#include "Profiler/Vendor/HcuProfiler.h"
#include "Context/Context.h"
#include "Driver/GPU/RoctracerApi.h"
#include "Profiler/Profiler.h"
#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <roctracer/roctracer_hip.h>
#include <set>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>

namespace proton {
namespace {
thread_local std::vector<RuntimeTraceEventKey> scopes;
thread_local bool inCollector = false;
struct CollectorGuard {
  bool previous = inCollector;
  CollectorGuard() { inCollector = true; }
  ~CollectorGuard() { inCollector = previous; }
};
void checkHip(hipError_t status, const char *op) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(op) + ": " +
                             hipGetErrorString(status));
}
class HcuProfiler final : public Profiler,
                          public OpInterface,
                          public Singleton<HcuProfiler> {
public:
  void configure(bool runtime, bool occupancy, bool resources) {
    runtimeEvents = runtime;
    queryOccupancy = occupancy;
    queryResources = resources;
  }
  VendorProfileArtifact collect(const SessionProfileMetadata &meta,
                                const VendorProfilePlan &plan) {
    std::lock_guard<std::mutex> lock(eventMutex);
    VendorProfileArtifact result;
    result.backend = "hcu";
    result.importer = "dtk_roctracer";
    result.requestedMetrics = plan.requested.vendorMetrics;
    result.enabledMetrics = plan.enabledVendorMetrics;
    result.degradeReasons = plan.degradeReasons;
    for (auto event : events) {
      auto it = launches.find(event.correlationId);
      if (it == launches.end() || !it->second.sessions.count(meta.sessionName))
        continue;
      const auto &launch = it->second;
      const bool kernel = !event.vendorMetrics.count("hcu.activity_kind");
      VendorMetricAssociation association;
      association.state = VendorMetricState::Collected;
      association.source = kernel ? "hcu_activity" : "hcu_runtime_activity";
      if (kernel) {
        event.scopeId = launch.scope.scopeId;
        association.metrics = launch.metrics;
        association.note = launch.error;
        association.metrics["hcu.kernel_duration_ns"] =
            double(event.endTimeNs - event.startTimeNs);
      }
      if (!kernel) {
        const auto kind =
            std::get<std::string>(event.vendorMetrics.at("hcu.activity_kind"));
        if (kind == "host_api")
          event.deviceId = launch.device;
        else if (!launch.memoryKind.empty()) {
          event.vendorMetrics["hcu.activity_kind"] = launch.memoryKind;
          event.vendorMetrics["hcu.bytes"] = launch.bytes;
          event.vendorMetrics["hcu.copy_kind"] = launch.copyKind;
          event.vendorMetrics["hcu.byte_count_source"] =
              std::string("hip_api_parameters");
        }
      }
      association.metrics.insert(event.vendorMetrics.begin(),
                                 event.vendorMetrics.end());
      association.runtimeEvent = std::move(event);
      result.associations.push_back(std::move(association));
    }
    for (const auto &cap : {"theoretical_occupancy", "resource_usage"}) {
      if (std::find(result.enabledMetrics.begin(), result.enabledMetrics.end(),
                    cap) == result.enabledMetrics.end())
        continue;
      const auto key = std::string(cap) == "resource_usage"
                           ? "hcu.registers_per_thread"
                           : "hcu.theoretical_occupancy_pct";
      size_t missing = 0, available = 0;
      for (const auto &a : result.associations)
        if (a.source == "hcu_activity") {
          if (a.metrics.count(key))
            ++available;
          else
            ++missing;
        }
      if (missing)
        result.degradeReasons.push_back(std::string(cap) + " unavailable for " +
                                        std::to_string(missing) +
                                        " kernels; see association notes");
      if (!available)
        result.enabledMetrics.erase(std::remove(result.enabledMetrics.begin(),
                                                result.enabledMetrics.end(),
                                                cap),
                                    result.enabledMetrics.end());
    }
    for (const auto &metric : plan.requested.vendorMetrics)
      if (metric.required &&
          std::find(result.enabledMetrics.begin(), result.enabledMetrics.end(),
                    metric.name) == result.enabledMetrics.end())
        throw std::runtime_error("Required HCU metric unavailable: " +
                                 metric.name);
    return result;
  }

private:
  struct Launch {
    RuntimeTraceEventKey scope;
    std::set<std::string> sessions;
    hipFunction_t function = nullptr;
    uint64_t threads = 0, shared = 0, epoch = 0;
    std::map<std::string, MetricValueType> metrics;
    std::string error;
    int device = 0;
    uint64_t begin = 0, bytes = 0, copyKind = 0;
    std::string apiName, memoryKind;
  };
  std::mutex eventMutex;
  std::unordered_map<uint64_t, Launch> launches;
  std::vector<RuntimeTraceEventKey> events;
  std::set<int> devices;
  uint64_t moduleEpoch = 0;
  bool runtimeEvents = false, queryOccupancy = false, queryResources = false;
  bool poolOpen = false, callbackEnabled = false, opsEnabled = false;
  std::string callbackError;
  void startOp(const Scope &scope) override {
    RuntimeTraceEventKey event;
    event.scopeId = scope.scopeId;
    event.opName = scope.name;
    scopes.push_back(std::move(event));
  }
  void stopOp(const Scope &) override {
    if (!scopes.empty())
      scopes.pop_back();
  }
  static void callback(uint32_t domain, uint32_t cid, const void *raw,
                       void *) noexcept {
    if (inCollector || !raw || domain != ACTIVITY_DOMAIN_HIP_API)
      return;
    auto &self = instance();
    try {
      CollectorGuard guard;
      const auto &data = *static_cast<const hip_api_data_t *>(raw);
      if (data.phase == ACTIVITY_API_PHASE_EXIT) {
        if (!self.runtimeEvents)
          return;
        roctracer_timestamp_t end = 0;
        roctracer::getTimestamp<true>(&end);
        std::lock_guard<std::mutex> lock(self.eventMutex);
        auto it = self.launches.find(data.correlation_id);
        if (it != self.launches.end() && end > it->second.begin) {
          RuntimeTraceEventKey event;
          event.correlationId = data.correlation_id;
          event.startTimeNs = it->second.begin;
          event.endTimeNs = end;
          event.opName = it->second.apiName;
          event.deviceId = it->second.device;
          event.streamId = 1000000ULL + uint64_t(syscall(SYS_gettid));
          event.vendorMetrics["hcu.activity_kind"] = std::string("host_api");
          event.vendorMetrics["hcu.timestamp_source"] =
              std::string("roctracer_callback_host");
          event.vendorMetrics["hcu.host_thread_id"] =
              uint64_t(syscall(SYS_gettid));
          self.events.push_back(std::move(event));
        }
        return;
      }
      if (data.phase != ACTIVITY_API_PHASE_ENTER)
        return;
      Launch launch;
      if (!scopes.empty())
        launch.scope = scopes.back();
      {
        std::shared_lock<std::shared_mutex> lock(self.mutex);
        for (auto *d : self.dataSet)
          launch.sessions.insert(d->getPath());
      }
      int device = -1;
      if (!launch.sessions.empty())
        checkHip(hipGetDevice(&device), "get launch device");
      launch.device = device;
      if (self.runtimeEvents && !launch.sessions.empty()) {
        roctracer::getTimestamp<true>(&launch.begin);
        const char *name = roctracer::getOpString(domain, cid, 0);
        launch.apiName = name ? name : "hip_api";
        if (cid == HIP_API_ID_hipMemcpyAsync) {
          launch.memoryKind = "memcpy";
          launch.bytes = data.args.hipMemcpyAsync.sizeBytes;
          launch.copyKind = data.args.hipMemcpyAsync.kind;
        } else if (cid == HIP_API_ID_hipMemcpy) {
          launch.memoryKind = "memcpy";
          launch.bytes = data.args.hipMemcpy.sizeBytes;
          launch.copyKind = data.args.hipMemcpy.kind;
        } else if (cid == HIP_API_ID_hipMemsetAsync) {
          launch.memoryKind = "memset";
          launch.bytes = data.args.hipMemsetAsync.sizeBytes;
        } else if (cid == HIP_API_ID_hipMemset) {
          launch.memoryKind = "memset";
          launch.bytes = data.args.hipMemset.sizeBytes;
        }
      }
      if (cid == HIP_API_ID_hipModuleLaunchKernel) {
        const auto &p = data.args.hipModuleLaunchKernel;
        launch.function = p.f;
        launch.threads = uint64_t(p.blockDimX) * p.blockDimY * p.blockDimZ;
        launch.shared = p.sharedMemBytes;
        launch.metrics = {{"hcu.grid_x", uint64_t(p.gridDimX)},
                          {"hcu.grid_y", uint64_t(p.gridDimY)},
                          {"hcu.grid_z", uint64_t(p.gridDimZ)},
                          {"hcu.block_x", uint64_t(p.blockDimX)},
                          {"hcu.block_y", uint64_t(p.blockDimY)},
                          {"hcu.block_z", uint64_t(p.blockDimZ)},
                          {"hcu.dynamic_shared_memory_bytes", launch.shared}};
      }
      std::lock_guard<std::mutex> lock(self.eventMutex);
      if (cid == HIP_API_ID_hipModuleUnload)
        ++self.moduleEpoch;
      launch.epoch = self.moduleEpoch;
      if (device >= 0)
        self.devices.insert(device);
      // Retain launch-time membership; delayed buffers must not include paused
      // work.
      if (!launch.sessions.empty())
        self.launches[data.correlation_id] = std::move(launch);
    } catch (const std::exception &e) {
      std::lock_guard<std::mutex> lock(self.eventMutex);
      self.callbackError = e.what();
    }
  }
  static void activity(const char *begin, const char *end, void *) noexcept {
    auto &self = instance();
    try {
      CollectorGuard guard;
      std::lock_guard<std::mutex> lock(self.eventMutex);
      auto *record = reinterpret_cast<const roctracer_record_t *>(begin);
      while (reinterpret_cast<const char *>(record) < end) {
        const bool kernel = record->domain == ACTIVITY_DOMAIN_HIP_OPS &&
                            record->op == HIP_OP_ID_DISPATCH;
        if (kernel || self.runtimeEvents) {
          if (record->end_ns <= record->begin_ns) {
            if (kernel)
              self.callbackError = "HCU kernel has invalid device timestamps";
          } else {
            RuntimeTraceEventKey event;
            event.correlationId = record->correlation_id;
            event.startTimeNs = record->begin_ns;
            event.endTimeNs = record->end_ns;
            if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
              event.streamId = 1000000ULL + record->thread_id;
              event.vendorMetrics["hcu.activity_kind"] =
                  std::string("host_api");
              event.vendorMetrics["hcu.host_thread_id"] =
                  uint64_t(record->thread_id);
            } else {
              event.deviceId = record->device_id;
              event.streamId = record->queue_id;
              if (!kernel) {
                event.vendorMetrics["hcu.activity_kind"] =
                    std::string("device_activity");
                event.vendorMetrics["hcu.operation"] = uint64_t(record->op);
                // The union is bytes only for copy activities, never for
                // dispatch.
                // DTK stores an opaque pointer in this union; use correlated
                // API arguments.
              }
            }
            const char *name =
                kernel ? record->kernel_name
                       : roctracer::getOpString(record->domain, record->op,
                                                record->kind);
            event.opName =
                name ? name : (kernel ? "hcu_kernel" : "hcu_activity");
            self.events.push_back(std::move(event));
          }
        }
        roctracer::getNextRecord<true>(record, &record);
      }
    } catch (const std::exception &e) {
      std::lock_guard<std::mutex> lock(self.eventMutex);
      self.callbackError = e.what();
    }
  }
  void resolveResources() {
    std::lock_guard<std::mutex> lock(eventMutex);
    for (const auto &event : events) {
      if (event.vendorMetrics.count("hcu.activity_kind"))
        continue;
      auto it = launches.find(event.correlationId);
      if (it == launches.end())
        continue;
      auto &launch = it->second;
      if (!launch.error.empty() ||
          launch.metrics.count("hcu.registers_per_thread"))
        continue;
      if (!launch.function || !launch.threads || launch.epoch != moduleEpoch) {
        launch.error = "Function/geometry unavailable, graph replay, or module "
                       "unloaded before query";
        continue;
      }
      auto status = hipSetDevice(event.deviceId);
      if (queryResources && status == hipSuccess) {
        for (const auto &[attr, name] :
             std::vector<std::pair<hipFunction_attribute, const char *>>{
                 {HIP_FUNC_ATTRIBUTE_NUM_REGS, "hcu.registers_per_thread"},
                 {HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                  "hcu.static_shared_memory_bytes"},
                 {HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                  "hcu.local_memory_bytes_per_thread"}}) {
          int value = 0;
          status = hipFuncGetAttribute(&value, attr, launch.function);
          if (status != hipSuccess)
            break;
          launch.metrics[name] = uint64_t(value);
        }
      }
      if (queryOccupancy && status == hipSuccess) {
        int blocks = 0, maxThreads = 0;
        status = hipDeviceGetAttribute(
            &maxThreads, hipDeviceAttributeMaxThreadsPerMultiProcessor,
            event.deviceId);
        if (status == hipSuccess)
          status = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, launch.function, launch.threads, launch.shared);
        if (status == hipSuccess && blocks > 0 && maxThreads > 0 &&
            uint64_t(blocks) * launch.threads <= uint64_t(maxThreads)) {
          launch.metrics["hcu.active_blocks_per_cu"] = uint64_t(blocks);
          launch.metrics["hcu.threads_per_block"] = launch.threads;
          launch.metrics["hcu.max_threads_per_cu"] = uint64_t(maxThreads);
          launch.metrics["hcu.theoretical_occupancy_pct"] =
              100.0 * blocks * launch.threads / maxThreads;
          launch.metrics["hcu.occupancy_source"] =
              std::string("hip_module_occupancy_api");
        } else if (status == hipSuccess)
          launch.error = "Invalid SDK occupancy limits";
      }
      if (status != hipSuccess)
        launch.error = hipGetErrorString(status);
    }
  }
  void cleanup() {
    if (opsEnabled) {
      roctracer::disableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
      opsEnabled = false;
    }
    if (callbackEnabled) {
      roctracer::disableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API);
      callbackEnabled = false;
    }
    if (poolOpen) {
      roctracer::closePool<true>();
      poolOpen = false;
    }
  }
  void doStart() override {
    CollectorGuard guard;
    {
      std::lock_guard<std::mutex> lock(eventMutex);
      events.clear();
      launches.clear();
      devices.clear();
      moduleEpoch = 0;
      callbackError.clear();
    }
    try {
      roctracer_properties_t properties{};
      properties.buffer_size = 8 * 1024 * 1024;
      properties.buffer_callback_fun = activity;
      roctracer::openPool<true>(&properties);
      poolOpen = true;
      roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API, callback,
                                            nullptr);
      callbackEnabled = true;
      roctracer::enableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
      opsEnabled = true;
      roctracer::start();
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void doFlush() override {
    CollectorGuard guard;
    int original = 0;
    checkHip(hipGetDevice(&original), "get current device");
    struct Restore {
      int id;
      ~Restore() { (void)hipSetDevice(id); }
    } restore{original};
    std::set<int> touched;
    {
      std::lock_guard<std::mutex> lock(eventMutex);
      touched = devices;
    }
    touched.insert(original);
    for (int device : touched) {
      checkHip(hipSetDevice(device), "select flush device");
      checkHip(hipDeviceSynchronize(), "synchronize flush device");
    }
    roctracer::flushActivity<true>();
    if (queryOccupancy || queryResources)
      resolveResources();
    std::lock_guard<std::mutex> lock(eventMutex);
    if (!callbackError.empty())
      throw std::runtime_error(callbackError);
  }
  void doStop() override {
    CollectorGuard guard;
    roctracer::stop();
    cleanup();
  }
};
class HcuImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override { return "dtk_roctracer"; }
  VendorProfileArtifact import(const SessionProfileMetadata &m,
                               const VendorProfilePlan &p) const override {
    return HcuProfiler::instance().collect(m, p);
  }
};
} // namespace
VendorProfilePlan
HcuAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  auto required = options.adapterOptions.find("require_metrics");
  bool require = false;
  if (required != options.adapterOptions.end()) {
    if (required->second != "true" && required->second != "false")
      throw std::invalid_argument("require_metrics must be true or false");
    require = required->second == "true";
  }
  if (require)
    for (auto &request : plan.requested.vendorMetrics)
      request.required = true;
  plan.runtimeBaseEnabled = options.runtimeBaseEnabled;
  const auto supported = getSupportedVendorMetrics();
  for (const auto &metric : plan.requested.vendorMetrics) {
    if (std::find(supported.begin(), supported.end(), metric.name) !=
        supported.end())
      plan.enabledVendorMetrics.push_back(metric.name);
    else {
      if (metric.required)
        throw std::runtime_error("HCU metric unsupported: " + metric.name);
      plan.disabledVendorMetrics.push_back(metric.name);
      plan.degradeReasons.push_back("HCU metric unsupported: " + metric.name);
    }
  }
  return plan;
}
void HcuAdapter::configureRuntimeProfiler(const std::string &,
                                          const VendorProfilePlan &plan) const {
  auto it = plan.requested.adapterOptions.find("runtime_events");
  auto runtime =
      it == plan.requested.adapterOptions.end() ? "false" : it->second;
  if (runtime != "true" && runtime != "false" && runtime != "1" &&
      runtime != "0")
    throw std::invalid_argument("runtime_events must be true, false, 1 or 0");
  it = plan.requested.adapterOptions.find("device_id");
  if (it != plan.requested.adapterOptions.end()) {
    int count = 0;
    checkHip(hipGetDeviceCount(&count), "get device count");
    if (it->second.empty() ||
        it->second.find_first_not_of("0123456789") != std::string::npos ||
        std::stoull(it->second) >= uint64_t(count))
      throw std::invalid_argument("device_id outside visible HCU device range");
  }
  const auto &enabled = plan.enabledVendorMetrics;
  HcuProfiler::instance().configure(
      runtime == "true" || runtime == "1",
      std::find(enabled.begin(), enabled.end(), "theoretical_occupancy") !=
          enabled.end(),
      std::find(enabled.begin(), enabled.end(), "resource_usage") !=
          enabled.end());
}
Profiler *HcuAdapter::getRuntimeProfiler() const {
  return &HcuProfiler::instance();
}
std::unique_ptr<VendorMetricsImporter> HcuAdapter::createImporter() const {
  return std::make_unique<HcuImporter>();
}
} // namespace proton
