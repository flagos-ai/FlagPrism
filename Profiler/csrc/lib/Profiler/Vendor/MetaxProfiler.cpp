// FlagPrism：native MACA activity collection; only device timestamps count as kernel time.
#include "Profiler/Vendor/MetaxProfiler.h"
#include "Context/Context.h"
#include "Profiler/Profiler.h"
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <set>
#include <stdexcept>
#include <mcr/mc_runtime_api.h>
#include <mcr/mcpti_runtime_cbid.h>
#include <tuple>
#include <mcpti/mcpti.h>
#include <unordered_map>

namespace proton {
namespace {
void check(MCptiResult result, const char *operation) {
  if (result != MCPTI_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             " failed: MCPTI status " +
                             std::to_string(static_cast<int>(result)));
}
#include "MetaxCounters.inc"
thread_local std::vector<RuntimeTraceEventKey> activeScopes;
thread_local bool insideCollector = false;
class MetaxProfiler final : public Profiler,
                              public OpInterface,
                              public Singleton<MetaxProfiler> {
public:
  void configure(bool runtimeEvents, bool occupancy, int device, std::vector<std::string> counterNames) {
    captureOccupancy = occupancy;
    captureRuntimeEvents = runtimeEvents;
    counters.names = std::move(counterNames);
    counters.selectedDevice = device;
  }
  VendorProfileArtifact collect(const SessionProfileMetadata &metadata,
                                const VendorProfilePlan &plan) {
    std::lock_guard<std::mutex> lock(eventsMutex);
    VendorProfileArtifact artifact;
    artifact.backend = metadata.backend;
    artifact.importer = "mcpti_activity";
    artifact.requestedMetrics = plan.requested.vendorMetrics;
    artifact.enabledMetrics = plan.enabledVendorMetrics;
    artifact.degradeReasons = plan.degradeReasons;
    for (auto event : events) {
      auto it = launches.find(event.correlationId);
      // Activity buffers arrive asynchronously. Attribute each launch to the
      // sessions active at API entry, not at buffer completion/import time.
      if (it == launches.end() ||
          !it->second.sessions.count(metadata.sessionName))
        continue;
      if (it->second.scope.scopeId) {
        event.scopeId = it->second.scope.scopeId;
        // Preserve the kernel symbol recorded by MCPTI.
      }
      VendorMetricAssociation association;
      const auto kindIt = event.vendorMetrics.find("metax.activity_kind");
      const bool kernel = kindIt == event.vendorMetrics.end();
      const std::string kind = kernel ? "kernel" : std::get<std::string>(kindIt->second);
      association.metrics["metax." + kind + "_duration_ns"] =
          double(event.endTimeNs - event.startTimeNs);
      for (const auto &[name, value] : event.vendorMetrics) {
        const bool launch = name.find("metax.grid_") == 0 ||
                            name.find("metax.block_") == 0;
        const auto capability = launch ? "launch_stats" : "resource_usage";
        if (!kernel || std::find(plan.enabledVendorMetrics.begin(),
                      plan.enabledVendorMetrics.end(), capability) !=
            plan.enabledVendorMetrics.end())
          association.metrics[name] = value;
      }
      if (kernel && captureOccupancy) {
        association.metrics.insert(it->second.occupancy.begin(), it->second.occupancy.end());
        if (!it->second.occupancyError.empty())
          association.note = "Theoretical occupancy unavailable: " + it->second.occupancyError;
      }
      if (kind == "runtime_api" || kind == "driver_api" || kind == "synchronization") {
        auto device = contextDevices.find(it->second.context);
        event.deviceId = device == contextDevices.end() ? metadata.device.id : device->second;
        association.metrics["metax.device_id_source"] = std::string(
            device == contextDevices.end() ? "session_default" : "callback_context");
      }
      association.runtimeEvent = std::move(event);
      association.source = kernel ? "mcpti_activity" : "mcpti_" + kind;
      if (!kernel)
        association.runtimeEvent.scopeId = 0; // Keep API/memory times out of kernel scopes.
      association.state = VendorMetricState::Collected;
      artifact.associations.push_back(std::move(association));
    }
    if (captureOccupancy) {
      size_t available = 0, missing = 0;
      for (const auto &association : artifact.associations) {
        if (association.source != "mcpti_activity") continue;
        if (association.metrics.count("metax.theoretical_occupancy_pct")) ++available;
        else ++missing;
      }
      if (missing)
        artifact.degradeReasons.push_back("Theoretical occupancy unavailable for " +
            std::to_string(missing) + " kernel activities; see association notes");
      if (!available)
        artifact.enabledMetrics.erase(std::remove(artifact.enabledMetrics.begin(),
            artifact.enabledMetrics.end(), "theoretical_occupancy"), artifact.enabledMetrics.end());
    }
    if (!counters.error.empty()) {
      artifact.degradeReasons.push_back("MCPTI hardware counters unavailable: " + counters.error);
      for (const auto &name : counters.names)
        artifact.enabledMetrics.erase(std::remove(artifact.enabledMetrics.begin(),
            artifact.enabledMetrics.end(), name), artifact.enabledMetrics.end());
    }
    for (const auto &range : counters.ranges) {
      VendorMetricAssociation association;
      association.source = "mcpti_hardware_counter";
      association.state = VendorMetricState::Collected;
      association.note = "Device counter range while the session is active; not per-kernel attribution. Throughput uses range duration in ns.";
      association.runtimeEvent.opName = "metax_counter_range";
      association.runtimeEvent.deviceId = range.device;
      association.runtimeEvent.streamId = 2000000;
      association.runtimeEvent.startTimeNs = range.start;
      association.runtimeEvent.endTimeNs = range.end;
      association.metrics = range.metrics;
      artifact.associations.push_back(std::move(association));
    }
    if (artifact.associations.empty())
      artifact.degradeReasons.push_back("MCPTI collected " + std::to_string(events.size()) +
          " kernel activities and " + std::to_string(launches.size()) +
          " API correlations, but none matched the active session");
    // Retain events across pause/resume; the next start begins a fresh capture.
    return artifact;
  }

private:
  // MACA 3.7 activity bookkeeping can race across host launch threads. Guard
  // complete API enter/exit pairs, retaining asynchronous device execution.
  std::recursive_mutex apiMutex;
  std::mutex eventsMutex;
  std::vector<RuntimeTraceEventKey> events;
  struct Launch {
    RuntimeTraceEventKey scope;
    std::set<std::string> sessions;
    MCcontext context = nullptr;
    mcFunction_t moduleFunction = nullptr;
    const void *runtimeFunction = nullptr;
    uint64_t threads = 0, sharedMemory = 0;
    std::map<std::string, MetricValueType> occupancy;
    std::string occupancyError;
    std::string apiName;
    uint32_t callbackId = 0, callbackDomain = 0;
    uint64_t moduleEpoch = 0;
  };
  std::unordered_map<uint32_t, Launch> launches;
  uint64_t moduleEpoch = 0;
  std::set<MCcontext> touchedContexts;
  std::unordered_map<MCcontext, uint32_t> contextDevices;
  MCpti_SubscriberHandle subscriber = nullptr;
  bool enabled = false;
  bool captureRuntimeEvents = false;
  bool captureOccupancy = false;
  MetaxCounters counters;
  std::vector<MCpti_ActivityKind> enabledKinds;
  std::string callbackError;
  void startOp(const Scope &scope) override {
    RuntimeTraceEventKey event;
    event.scopeId = scope.scopeId;
    event.opName = scope.name;
    activeScopes.push_back(std::move(event));
  }
  void stopOp(const Scope &) override {
    if (!activeScopes.empty())
      activeScopes.pop_back();
  }
  static void callback(void *, MCpti_CallbackDomain domain, uint32_t cbid,
                       const void *raw) {
    if (!raw || insideCollector)
      return;
    auto *data = static_cast<const MCpti_CallbackData *>(raw);
    auto &self = instance();
    if (data->callbackSite == MCPTI_API_EXIT) {
      self.apiMutex.unlock();
      return;
    }
    if (data->callbackSite != MCPTI_API_ENTER)
      return;
    self.apiMutex.lock();
    Launch launch;
    // SDK queries are deferred until flush; callback-time device queries may
    // reenter the vendor runtime while another thread is launching a kernel.
    launch.context = data->context;
    launch.apiName = data->functionName ? data->functionName : "unknown";
    launch.callbackId = cbid;
    launch.callbackDomain = domain;
    if (self.captureOccupancy && data->functionParams) {
      if (launch.apiName == "mcModuleLaunchKernel" && cbid == MCPTI_RUNTIME_TRACE_CBID_mcModuleLaunchKernel) {
        const auto *params = static_cast<const mcModuleLaunchKernel_params *>(data->functionParams);
        launch.moduleFunction = params->f;
        launch.threads = uint64_t(params->blockDimX) * params->blockDimY * params->blockDimZ;
        launch.sharedMemory = params->sharedMemBytes;
      } else if (launch.apiName == "mcLaunchKernel" && cbid == MCPTI_RUNTIME_TRACE_CBID_mcLaunchKernel) {
        const auto *params = static_cast<const mcLaunchKernel_params *>(data->functionParams);
        launch.runtimeFunction = params->function_address;
        launch.threads = uint64_t(params->dimBlocks.x) * params->dimBlocks.y * params->dimBlocks.z;
        launch.sharedMemory = params->sharedMemBytes;
      }
    }
    if (!activeScopes.empty())
      launch.scope = activeScopes.back();
    {
      // Keep Data alive while copying its identity: unregisterData takes the
      // exclusive lock before a finalized session can destroy these objects.
      std::shared_lock<std::shared_mutex> lock(self.mutex);
      for (auto *data : self.dataSet)
        launch.sessions.insert(data->getPath());
    }
    std::lock_guard<std::mutex> lock(self.eventsMutex);
    // A module unload may invalidate an already-captured function handle.
    // Conservatively skip unresolved module queries from earlier epochs.
    if (launch.apiName == "mcModuleUnload") ++self.moduleEpoch;
    launch.moduleEpoch = self.moduleEpoch;
    if (!launch.sessions.empty() && launch.context)
      self.touchedContexts.insert(launch.context);
    self.launches[data->correlationId] = std::move(launch);
  }
  static void request(uint8_t **buffer, size_t *size, size_t *maxRecords) {
    *size = 8 * 1024 * 1024;
    *maxRecords = 0;
    *buffer = static_cast<uint8_t *>(std::malloc(*size));
    if (!*buffer)
      *size = 0;
  }
  static void complete(MCcontext context, uint32_t streamId, uint8_t *buffer, size_t, size_t valid) {
    if (!valid) {
      std::free(buffer);
      return;
    }
    auto &self = instance();
    std::lock_guard<std::mutex> lock(self.eventsMutex);
    MCpti_Activity *record = nullptr;
    MCptiResult result;
    while ((result = mcptiActivityGetNextRecord(buffer, valid, &record)) ==
           MCPTI_SUCCESS) {
      if (record->kind != MCPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        RuntimeTraceEventKey event;
        std::string kind;
        if (record->kind == MCPTI_ACTIVITY_KIND_MEMCPY) {
          auto *copy = reinterpret_cast<MCpti_ActivityMemcpy5 *>(record);
          kind = "memcpy";
          event.startTimeNs = copy->start; event.endTimeNs = copy->end;
          event.deviceId = copy->deviceId; event.streamId = copy->streamId;
          event.correlationId = copy->correlationId;
          event.vendorMetrics["metax.bytes"] = uint64_t(copy->bytes);
          event.vendorMetrics["metax.copy_kind"] = uint64_t(copy->copyKind);
        } else if (record->kind == MCPTI_ACTIVITY_KIND_MEMSET) {
          auto *set = reinterpret_cast<MCpti_ActivityMemset4 *>(record);
          kind = "memset";
          event.startTimeNs = set->start; event.endTimeNs = set->end;
          event.deviceId = set->deviceId; event.streamId = set->streamId;
          event.correlationId = set->correlationId;
          event.vendorMetrics["metax.bytes"] = uint64_t(set->bytes);
          event.vendorMetrics["metax.value"] = uint64_t(set->value);
        } else if (record->kind == MCPTI_ACTIVITY_KIND_RUNTIME ||
                   record->kind == MCPTI_ACTIVITY_KIND_DRIVER) {
          auto *api = reinterpret_cast<MCpti_ActivityAPI *>(record);
          kind = record->kind == MCPTI_ACTIVITY_KIND_RUNTIME ? "runtime_api" : "driver_api";
          event.startTimeNs = api->start; event.endTimeNs = api->end;
          event.correlationId = api->correlationId;
          event.streamId = 1000000ULL + api->threadId;
          const char *name = nullptr;
          mcptiGetCallbackName(record->kind == MCPTI_ACTIVITY_KIND_RUNTIME ?
              MCPTI_CB_DOMAIN_RUNTIME_API : MCPTI_CB_DOMAIN_DRIVER_API, api->cbid, &name);
          event.opName = name ? name : kind;
          event.vendorMetrics["metax.return_value"] = uint64_t(api->returnValue);
          event.vendorMetrics["metax.host_thread_id"] = uint64_t(api->threadId);
        } else if (record->kind == MCPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
          auto *sync = reinterpret_cast<MCpti_ActivitySynchronization *>(record);
          kind = "synchronization";
          event.startTimeNs = sync->start; event.endTimeNs = sync->end;
          event.correlationId = sync->correlationId; event.streamId = sync->streamId;
          event.vendorMetrics["metax.synchronization_type"] = uint64_t(sync->type);
        } else {
          continue;
        }
        // Zero-duration API calls are valid but cannot form a timed interval.
        if (!event.startTimeNs || event.endTimeNs <= event.startTimeNs)
          continue;
        if (event.opName.empty()) event.opName = "metax_" + kind;
        event.vendorMetrics["metax.activity_kind"] = kind;
        self.events.push_back(std::move(event));
        continue;
      }
      auto *kernel = reinterpret_cast<MCpti_ActivityKernel8 *>(record);
      if (!kernel->start || kernel->end <= kernel->start) {
        self.callbackError = "MCPTI kernel has invalid device timestamps";
        continue;
      }
      RuntimeTraceEventKey event;
      event.opName = kernel->name ? kernel->name : "metax_kernel";
      event.startTimeNs = kernel->start;
      event.endTimeNs = kernel->end;
      event.deviceId = kernel->deviceId;
      event.streamId = kernel->streamId;
      event.correlationId = kernel->correlationId;
      event.taskId = kernel->gridId;
      event.vendorMetrics = {
          {"metax.grid_x", uint64_t(kernel->gridX)},
          {"metax.grid_y", uint64_t(kernel->gridY)},
          {"metax.grid_z", uint64_t(kernel->gridZ)},
          {"metax.block_x", uint64_t(kernel->blockX)},
          {"metax.block_y", uint64_t(kernel->blockY)},
          {"metax.block_z", uint64_t(kernel->blockZ)},
          {"metax.registers_per_thread", uint64_t(kernel->registersPerThread)},
          {"metax.static_shared_memory_bytes", uint64_t(kernel->staticSharedMemory)},
          {"metax.dynamic_shared_memory_bytes", uint64_t(kernel->dynamicSharedMemory)},
          {"metax.local_memory_bytes_per_thread", uint64_t(kernel->localMemoryPerThread)},
      };
      self.events.push_back(std::move(event));
    }
    if (result != MCPTI_ERROR_MAX_LIMIT_REACHED)
      self.callbackError =
          "MCPTI could not decode activity buffer: " + std::to_string(result);
    size_t dropped = 0;
    auto droppedStatus = mcptiActivityGetNumDroppedRecords(context, streamId, &dropped);
    if (droppedStatus != MCPTI_SUCCESS || dropped)
      self.callbackError = "MCPTI lost activity records: " + std::to_string(dropped);
    std::free(buffer);
  }
  void resolveOccupancy() {
    if (!captureOccupancy) return;
    struct Query { uint32_t correlation, device; Launch launch; };
    std::vector<Query> queries;
    {
      std::lock_guard<std::mutex> lock(eventsMutex);
      for (const auto &event : events) {
        if (event.vendorMetrics.count("metax.activity_kind")) continue;
        auto it = launches.find(event.correlationId);
        if (it != launches.end() && !it->second.sessions.empty() &&
            it->second.occupancy.empty() && it->second.occupancyError.empty()) {
          if (it->second.moduleFunction && it->second.moduleEpoch != moduleEpoch) {
            it->second.occupancyError = "module unload observed before query; function lifetime unavailable";
            continue;
          }
          queries.push_back({uint32_t(event.correlationId), uint32_t(event.deviceId), it->second});
        }
      }
    }
    // Cache only within this flush. An unloaded/reused function handle cannot
    // silently reuse a previous flush's result. Failed SDK queries stay explicit.
    using Key = std::tuple<uint32_t, uintptr_t, bool, uint64_t, uint64_t>;
    std::map<Key, std::map<std::string, MetricValueType>> cache;
    for (const auto &query : queries) {
      const auto &launch = query.launch;
      std::string error;
      std::map<std::string, MetricValueType> metrics;
      const bool module = launch.moduleFunction != nullptr;
      const auto function = module ? reinterpret_cast<uintptr_t>(launch.moduleFunction)
                                   : reinterpret_cast<uintptr_t>(launch.runtimeFunction);
      const Key key{query.device, function, module, launch.threads, launch.sharedMemory};
      auto cached = cache.find(key);
      if (cached != cache.end()) metrics = cached->second;
      else if (!function || !launch.threads || launch.threads > INT32_MAX)
        error = "launch function/geometry unavailable: " + launch.apiName +
                " domain=" + std::to_string(launch.callbackDomain) +
                " cbid=" + std::to_string(launch.callbackId);
      else {
        int blocks = 0;
        mcDeviceProp_t properties{};
        auto status = mcSetDevice(query.device);
        if (status == mcSuccess) status = mcGetDeviceProperties(&properties, query.device);
        if (status == mcSuccess)
          status = module ? mcModuleOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, launch.moduleFunction, int(launch.threads), launch.sharedMemory)
                          : mcOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, launch.runtimeFunction, int(launch.threads), launch.sharedMemory);
        if (status != mcSuccess)
          error = "MACA occupancy API status " + std::to_string(int(status));
        else if (blocks <= 0 || properties.maxThreadsPerMultiProcessor <= 0 ||
                 uint64_t(blocks) * launch.threads > uint64_t(properties.maxThreadsPerMultiProcessor))
          error = "invalid SDK occupancy limits";
        else {
          metrics = {
            {"metax.active_blocks_per_sm", uint64_t(blocks)},
            {"metax.threads_per_block", launch.threads},
            {"metax.max_threads_per_sm", uint64_t(properties.maxThreadsPerMultiProcessor)},
            {"metax.theoretical_occupancy_pct", 100.0 * blocks * launch.threads / properties.maxThreadsPerMultiProcessor},
            {"metax.occupancy_source", std::string(module ? "maca_module_occupancy_api" : "maca_runtime_occupancy_api")},
            {"metax.occupancy_dynamic_shared_memory_bytes", launch.sharedMemory},
          };
          cache[key] = metrics;
        }
      }
      std::lock_guard<std::mutex> lock(eventsMutex);
      auto &result = launches[query.correlation];
      result.occupancy = std::move(metrics);
      result.occupancyError = std::move(error);
    }
  }
  void doStart() override {
    {
      std::lock_guard<std::mutex> lock(eventsMutex);
      events.clear();
      launches.clear();
      moduleEpoch = 0;
      touchedContexts.clear();
      contextDevices.clear();
      callbackError.clear();
    }
    check(mcptiActivityRegisterCallbacks(request, complete),
          "register activity callbacks");
    check(mcptiSubscribe(&subscriber, callback, nullptr),
          "subscribe callbacks");
    try {
      check(mcptiEnableDomain(1, subscriber, MCPTI_CB_DOMAIN_RUNTIME_API),
            "enable runtime callbacks");
      check(mcptiEnableDomain(1, subscriber, MCPTI_CB_DOMAIN_DRIVER_API),
            "enable driver callbacks");
      enabledKinds.clear();
      std::vector<MCpti_ActivityKind> kinds = {MCPTI_ACTIVITY_KIND_CONCURRENT_KERNEL};
      if (captureRuntimeEvents) {
        kinds.insert(kinds.end(), {MCPTI_ACTIVITY_KIND_MEMCPY, MCPTI_ACTIVITY_KIND_MEMSET,
          MCPTI_ACTIVITY_KIND_RUNTIME, MCPTI_ACTIVITY_KIND_DRIVER,
          MCPTI_ACTIVITY_KIND_SYNCHRONIZATION});
      }
      for (auto kind : kinds) {
        check(mcptiActivityEnable(kind), "enable MCPTI activity");
        enabledKinds.push_back(kind);
      }
      enabled = true;
      counters.prepare();
    } catch (...) {
      for (auto kind : enabledKinds) mcptiActivityDisable(kind);
      enabledKinds.clear();
      mcptiUnsubscribe(subscriber);
      subscriber = nullptr;
      throw;
    }
  }
  void doFlush() override {
    if (!enabled)
      return;
    // Synchronize every device seen in callbacks, restoring the caller device.
    struct FlushGuard {
      bool previous = insideCollector;
      int original = -1;
      FlushGuard() {
        insideCollector = true;
        if (mcGetDevice(&original) != mcSuccess) {
          insideCollector = previous;
          throw std::runtime_error("mcGetDevice failed during MCPTI flush");
        }
      }
      ~FlushGuard() {
        if (original >= 0) mcSetDevice(original);
        insideCollector = previous;
      }
    } guard;
    std::set<int> devices;
    std::set<MCcontext> contexts;
    {
      std::lock_guard<std::mutex> lock(eventsMutex);
      contexts = touchedContexts;
    }
    for (auto context : contexts) {
      uint32_t device = 0;
      check(mcptiGetDeviceId(context, &device), "query activity context device");
      devices.insert(static_cast<int>(device));
      std::lock_guard<std::mutex> lock(eventsMutex);
      contextDevices[context] = device;
    }
    devices.insert(guard.original);
    for (int device : devices) {
      if (mcSetDevice(device) != mcSuccess || mcDeviceSynchronize() != mcSuccess)
        throw std::runtime_error("device synchronization failed during MCPTI flush");
    }
    // A flush before registerData begins a range; a flush before unregisterData
    // ends it. Paused work is therefore excluded without replaying user kernels.
    if (!counters.names.empty()) {
      if (mcSetDevice(counters.deviceId()) != mcSuccess)
        throw std::runtime_error("counter device selection failed");
      if (getDataSet().empty()) counters.begin();
      else counters.finish();
    }
    check(mcptiActivityFlushAll(MCPTI_ACTIVITY_FLAG_FLUSH_FORCED),
          "flush activities");
    resolveOccupancy();
    std::lock_guard<std::mutex> lock(eventsMutex);
    if (!callbackError.empty())
      throw std::runtime_error(callbackError);
  }
  void doStop() override {
    counters.release();
    if (enabled) {
      for (auto kind : enabledKinds)
        check(mcptiActivityDisable(kind), "disable MCPTI activity");
      enabledKinds.clear();
      enabled = false;
    }
    if (subscriber) {
      check(mcptiUnsubscribe(subscriber), "unsubscribe callbacks");
      subscriber = nullptr;
    }
    // Keep SDK attachment alive across sessions. On MACA 3.7, detaching with
    // mcptiFinalize can leave activity recording enabled but API callbacks
    // silent on the next subscription, losing session attribution.
  }
};
class MetaxImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override { return "mcpti_activity"; }
  VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                               const VendorProfilePlan &plan) const override {
    return MetaxProfiler::instance().collect(metadata, plan);
  }
};
} // namespace
VendorProfilePlan
MetaxAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = options.runtimeBaseEnabled;
  for (const auto &metric : options.vendorMetrics) {
    const auto supported = getSupportedVendorMetrics();
    if (std::find(supported.begin(), supported.end(), metric.name) != supported.end()) {
      plan.enabledVendorMetrics.push_back(metric.name);
      continue;
    }
    if (metric.required)
      throw std::runtime_error(
          "Metax MCPTI hardware counter unsupported: " + metric.name);
    plan.disabledVendorMetrics.push_back(metric.name);
    plan.degradeReasons.push_back("unsupported hardware counter: " +
                                  metric.name);
  }
  return plan;
}
void MetaxAdapter::configureRuntimeProfiler(const std::string &,
                                             const VendorProfilePlan &plan) const {
  auto it = plan.requested.adapterOptions.find("runtime_events");
  const auto value = it == plan.requested.adapterOptions.end() ? "false" : it->second;
  if (value != "true" && value != "false" && value != "1" && value != "0")
    throw std::invalid_argument("runtime_events must be true, false, 1 or 0");
  std::vector<std::string> hardware;
  for (const auto &name : plan.enabledVendorMetrics)
    if (name != "launch_stats" && name != "resource_usage" && name != "kernel_duration" && name != "theoretical_occupancy")
      hardware.push_back(name);
  const bool occupancy = std::find(plan.enabledVendorMetrics.begin(),
      plan.enabledVendorMetrics.end(), "theoretical_occupancy") != plan.enabledVendorMetrics.end();
  // FlagPrism：validate explicit device selection before acquiring SDK resources.
  int device = -1;
  auto selected = plan.requested.adapterOptions.find("device_id");
  if (selected != plan.requested.adapterOptions.end()) {
    const auto &text = selected->second;
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("device_id must be a non-negative integer");
    size_t consumed = 0;
    uint64_t parsed = 0;
    try {
      parsed = std::stoull(text, &consumed);
    } catch (const std::exception &) {
      throw std::invalid_argument("device_id is outside the supported integer range");
    }
    int count = 0;
    if (consumed != text.size() || mcGetDeviceCount(&count) != mcSuccess || parsed >= uint64_t(count))
      throw std::invalid_argument("device_id is outside the visible MetaX device range");
    device = static_cast<int>(parsed);
  }
  MetaxProfiler::instance().configure(value == "true" || value == "1", occupancy, device, std::move(hardware));
  // FlagPrism end
}
Profiler *MetaxAdapter::getRuntimeProfiler() const {
  return &MetaxProfiler::instance();
}
std::unique_ptr<VendorMetricsImporter> MetaxAdapter::createImporter() const {
  return std::make_unique<MetaxImporter>();
}
} // namespace proton

// FlagPrism end
