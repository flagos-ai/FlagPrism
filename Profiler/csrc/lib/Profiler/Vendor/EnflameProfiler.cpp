#include "Profiler/Vendor/EnflameProfiler.h"
#include "Context/Context.h"
#include "Profiler/Profiler.h"
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tops_runtime.h>
#include <topspti_activity.h>
#include <topspti_callbacks.h>
#include <unordered_map>
#include <vector>

namespace proton {
namespace {
void check(TopsptiResult result, const char *operation) {
  if (result != TOPSPTI_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             " failed: TOPSPTI status " +
                             std::to_string(static_cast<int>(result)));
}

// The TOPS headers only provide types; libtopspti and libtopsrt are loaded at
// runtime so the profiler builds without a TopsPlatform installation.
struct EnflameTopsptiApi {
  using RegisterCallbacks = TopsptiResult (*)(
      Topspti_BuffersCallbackRequestFunc, Topspti_BuffersCallbackCompleteFunc);
  using ActivityEnable = TopsptiResult (*)(Topspti_ActivityKind);
  using ActivityDisable = TopsptiResult (*)(Topspti_ActivityKind);
  using ActivityFlushAll = TopsptiResult (*)(uint32_t);
  using ActivityGetNextRecord = TopsptiResult (*)(uint8_t *, size_t,
                                                  Topspti_Activity **);
  using ActivityGetNumDroppedRecords = TopsptiResult (*)(size_t *);
  using Subscribe = TopsptiResult (*)(Topspti_SubscriberHandle *,
                                      Topspti_CallbackFunc, void *);
  using Unsubscribe = TopsptiResult (*)(Topspti_SubscriberHandle);
  using EnableDomain = TopsptiResult (*)(uint32_t, Topspti_SubscriberHandle,
                                         Topspti_CallbackDomain);
  using DeviceSynchronize = topsError_t (*)();

  void *library = nullptr;
  void *runtimeLibrary = nullptr;
  RegisterCallbacks registerCallbacks = nullptr;
  ActivityEnable activityEnable = nullptr;
  ActivityDisable activityDisable = nullptr;
  ActivityFlushAll activityFlushAll = nullptr;
  ActivityGetNextRecord activityGetNextRecord = nullptr;
  ActivityGetNumDroppedRecords activityGetNumDroppedRecords = nullptr;
  Subscribe subscribe = nullptr;
  Unsubscribe unsubscribe = nullptr;
  EnableDomain enableDomain = nullptr;
  DeviceSynchronize deviceSynchronize = nullptr;

  // TOPSPTI may still own activity buffers at process teardown and libtopsrt
  // is shared with the Triton launcher, so neither library is dlclose'd.
  bool load() {
    if (!library) {
      library = open("libtopspti.so", "/extras/TOPSPTI/lib64/");
      if (library) {
        registerCallbacks = reinterpret_cast<RegisterCallbacks>(
            dlsym(library, "topsptiActivityRegisterCallbacks"));
        activityEnable = reinterpret_cast<ActivityEnable>(
            dlsym(library, "topsptiActivityEnable"));
        activityDisable = reinterpret_cast<ActivityDisable>(
            dlsym(library, "topsptiActivityDisable"));
        activityFlushAll = reinterpret_cast<ActivityFlushAll>(
            dlsym(library, "topsptiActivityFlushAll"));
        activityGetNextRecord = reinterpret_cast<ActivityGetNextRecord>(
            dlsym(library, "topsptiActivityGetNextRecord"));
        activityGetNumDroppedRecords =
            reinterpret_cast<ActivityGetNumDroppedRecords>(
                dlsym(library, "topsptiActivityGetNumDroppedRecords"));
        subscribe =
            reinterpret_cast<Subscribe>(dlsym(library, "topsptiSubscribe"));
        unsubscribe =
            reinterpret_cast<Unsubscribe>(dlsym(library, "topsptiUnsubscribe"));
        enableDomain = reinterpret_cast<EnableDomain>(
            dlsym(library, "topsptiEnableDomain"));
      }
    }
    if (!runtimeLibrary) {
      runtimeLibrary = open("libtopsrt.so.1", "/lib/");
      if (runtimeLibrary) {
        deviceSynchronize = reinterpret_cast<DeviceSynchronize>(
            dlsym(runtimeLibrary, "topsDeviceSynchronize"));
      }
    }
    return registerCallbacks && activityEnable && activityDisable &&
           activityFlushAll && activityGetNextRecord &&
           activityGetNumDroppedRecords && subscribe && unsubscribe &&
           enableDomain && deviceSynchronize;
  }

private:
  // The bare soname comes first so an already loaded library is reused.
  static void *open(const char *soname, const char *subdir) {
    std::vector<std::string> candidates = {soname};
    for (const char *env : {"TOPS_HOME", "CAPS_PATH"}) {
      const char *root = std::getenv(env);
      if (root && *root) {
        candidates.push_back(std::string(root) + subdir + soname);
      }
    }
    candidates.push_back(std::string("/opt/tops") + subdir + soname);
    for (const auto &candidate : candidates) {
      if (void *handle = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY)) {
        return handle;
      }
    }
    return nullptr;
  }
};

EnflameTopsptiApi &topsptiApi() {
  static EnflameTopsptiApi api;
  return api;
}
thread_local std::vector<RuntimeTraceEventKey> activeScopes;
class EnflameProfiler final : public Profiler,
                              public OpInterface,
                              public Singleton<EnflameProfiler> {
public:
  VendorProfileArtifact collect(const SessionProfileMetadata &metadata,
                                const VendorProfilePlan &plan) {
    std::lock_guard<std::mutex> lock(eventsMutex);
    VendorProfileArtifact artifact;
    artifact.backend = metadata.backend;
    artifact.importer = "topspti_activity";
    artifact.requestedMetrics = plan.requested.vendorMetrics;
    for (auto event : events) {
      auto it = launches.find(event.correlationId);
      // Activity buffers arrive asynchronously. Attribute each launch to the
      // sessions active at API entry, not at buffer completion/import time.
      if (it == launches.end() ||
          !it->second.sessions.count(metadata.sessionName))
        continue;
      if (it->second.scope.scopeId) {
        event.scopeId = it->second.scope.scopeId;
        event.opName = it->second.scope.opName;
      }
      VendorMetricAssociation association;
      association.runtimeEvent = std::move(event);
      association.source = "topspti_activity";
      association.state = VendorMetricState::Collected;
      artifact.associations.push_back(std::move(association));
    }
    // Retain events across pause/resume; the next start begins a fresh capture.
    return artifact;
  }

private:
  std::mutex eventsMutex;
  std::vector<RuntimeTraceEventKey> events;
  struct Launch {
    RuntimeTraceEventKey scope;
    std::set<std::string> sessions;
  };
  std::unordered_map<uint32_t, Launch> launches;
  Topspti_SubscriberHandle subscriber = nullptr;
  bool enabled = false;
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
  static void callback(void *, Topspti_CallbackDomain, Topspti_CallbackId,
                       const void *raw) {
    if (!raw)
      return;
    auto *data = static_cast<const Topspti_CallbackData *>(raw);
    if (data->callbackSite != TOPSPTI_API_ENTER)
      return;
    auto &self = instance();
    Launch launch;
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
    self.launches[data->correlationId] = std::move(launch);
  }
  static void request(uint8_t **buffer, size_t *size, size_t *maxRecords) {
    *size = 8 * 1024 * 1024;
    *maxRecords = 0;
    *buffer = static_cast<uint8_t *>(std::malloc(*size));
    if (!*buffer)
      *size = 0;
  }
  static void complete(uint8_t *buffer, size_t, size_t valid) {
    if (!valid) {
      std::free(buffer);
      return;
    }
    auto &self = instance();
    std::lock_guard<std::mutex> lock(self.eventsMutex);
    Topspti_Activity *record = nullptr;
    TopsptiResult result;
    auto &api = topsptiApi();
    while ((result = api.activityGetNextRecord(buffer, valid, &record)) ==
           TOPSPTI_SUCCESS) {
      if (record->kind != TOPSPTI_ACTIVITY_KIND_KERNEL)
        continue;
      auto *kernel = reinterpret_cast<Topspti_ActivityKernel *>(record);
      if (!kernel->start || kernel->end <= kernel->start) {
        self.callbackError = "TOPSPTI kernel has invalid device timestamps";
        continue;
      }
      RuntimeTraceEventKey event;
      event.opName = kernel->name ? kernel->name : "enflame_kernel";
      event.startTimeNs = kernel->start;
      event.endTimeNs = kernel->end;
      event.deviceId = kernel->deviceId;
      event.streamId = kernel->streamId;
      event.correlationId = kernel->correlationId;
      event.taskId = kernel->gridId;
      self.events.push_back(std::move(event));
    }
    if (result != TOPSPTI_ERROR_MAX_LIMIT_REACHED)
      self.callbackError =
          "TOPSPTI could not decode activity buffer: " + std::to_string(result);
    std::free(buffer);
  }
  void doStart() override {
    {
      std::lock_guard<std::mutex> lock(eventsMutex);
      events.clear();
      launches.clear();
      callbackError.clear();
    }
    auto &api = topsptiApi();
    if (!api.load())
      throw std::runtime_error(
          "failed to load TOPSPTI from libtopspti.so and libtopsrt.so.1");
    check(api.registerCallbacks(request, complete),
          "register activity callbacks");
    check(api.subscribe(&subscriber, callback, nullptr), "subscribe callbacks");
    try {
      check(api.enableDomain(1, subscriber, TOPSPTI_CB_DOMAIN_RUNTIME_API),
            "enable runtime callbacks");
      check(api.enableDomain(1, subscriber, TOPSPTI_CB_DOMAIN_DRIVER_API),
            "enable driver callbacks");
      check(api.activityEnable(TOPSPTI_ACTIVITY_KIND_KERNEL),
            "enable kernel activities");
      enabled = true;
    } catch (...) {
      api.unsubscribe(subscriber);
      subscriber = nullptr;
      throw;
    }
  }
  void doFlush() override {
    if (!enabled)
      return;
    auto &api = topsptiApi();
    if (api.deviceSynchronize() != topsSuccess)
      throw std::runtime_error(
          "topsDeviceSynchronize failed during TOPSPTI flush");
    check(api.activityFlushAll(TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED),
          "flush activities");
    size_t dropped = 0;
    check(api.activityGetNumDroppedRecords(&dropped), "get dropped activities");
    std::lock_guard<std::mutex> lock(eventsMutex);
    if (dropped)
      throw std::runtime_error("TOPSPTI dropped " + std::to_string(dropped) +
                               " records");
    if (!callbackError.empty())
      throw std::runtime_error(callbackError);
  }
  void doStop() override {
    if (enabled) {
      check(topsptiApi().activityDisable(TOPSPTI_ACTIVITY_KIND_KERNEL),
            "disable kernel activities");
      enabled = false;
    }
    if (subscriber) {
      check(topsptiApi().unsubscribe(subscriber), "unsubscribe callbacks");
      subscriber = nullptr;
    }
  }
};
class EnflameImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override { return "topspti_activity"; }
  VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                               const VendorProfilePlan &plan) const override {
    return EnflameProfiler::instance().collect(metadata, plan);
  }
};
} // namespace
VendorProfilePlan
EnflameAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = options.runtimeBaseEnabled;
  for (const auto &metric : options.vendorMetrics) {
    if (metric.required)
      throw std::runtime_error(
          "Enflame TOPSPTI hardware counter unsupported: " + metric.name);
    plan.disabledVendorMetrics.push_back(metric.name);
    plan.degradeReasons.push_back("unsupported hardware counter: " +
                                  metric.name);
  }
  return plan;
}
Profiler *EnflameAdapter::getRuntimeProfiler() const {
  return &EnflameProfiler::instance();
}
std::unique_ptr<VendorMetricsImporter> EnflameAdapter::createImporter() const {
  return std::make_unique<EnflameImporter>();
}
} // namespace proton
