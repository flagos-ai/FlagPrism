#include "Profiler/Cupti/CuptiProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiPCSampling.h"
#include "Profiler/Cupti/NvidiaHardwareCounters.h"
#include "Utility/Map.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace proton {

template <>
thread_local GPUProfiler<CuptiProfiler>::ThreadState
    GPUProfiler<CuptiProfiler>::threadState(CuptiProfiler::instance());

template <>
thread_local std::deque<size_t>
    GPUProfiler<CuptiProfiler>::Correlation::externIdQueue{};

namespace {

#if CUPTI_API_VERSION >= 17
// FlagPrism: CUPTI API 17 introduced Kernel9 activity records. CUDA 11.6
// reports API 16 and still exposes Kernel5, so select the record layout by
// CUPTI API version rather than by the CUDA major version alone.
using CuptiKernelActivity = CUpti_ActivityKernel9;
#else
using CuptiKernelActivity = CUpti_ActivityKernel5;
#endif

#if CUPTI_API_VERSION >= 26
// FlagPrism: CUPTI 12.8 introduced Memcpy6. Older supported toolkits expose
// the same fields needed here through Memcpy5, so keep the importer source
// compatible with CUDA 11.6 through current CUPTI headers.
using CuptiMemcpyActivity = CUpti_ActivityMemcpy6;
#else
using CuptiMemcpyActivity = CUpti_ActivityMemcpy5;
#endif
using CuptiMemsetActivity = CUpti_ActivityMemset4;

std::shared_ptr<Metric> convertActivityToMetric(CUpti_Activity *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    auto *kernel = reinterpret_cast<CuptiKernelActivity *>(activity);
    if (kernel->start < kernel->end) {
      metric = std::make_shared<KernelMetric>(
          static_cast<uint64_t>(kernel->start),
          static_cast<uint64_t>(kernel->end), 1,
          static_cast<uint64_t>(kernel->deviceId),
          static_cast<uint64_t>(DeviceType::CUDA),
          static_cast<uint64_t>(kernel->streamId));
    } // else: not a valid kernel activity
    break;
  }
  default:
    break;
  }
  return metric;
}

uint32_t
processActivityKernel(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                      CuptiProfiler::ApiExternIdSet &apiExternIds,
                      std::set<Data *> &dataSet, CUpti_Activity *activity) {
  // Support CUDA >= 11.0
  auto *kernel = reinterpret_cast<CuptiKernelActivity *>(activity);
  auto correlationId = kernel->correlationId;
  if (/*Not a valid context*/ !corrIdToExternId.contain(correlationId))
    return correlationId;
  auto [parentId, numInstances] = corrIdToExternId.at(correlationId);
  if (kernel->graphId == 0) {
    // Non-graph kernels
    for (auto *data : dataSet) {
      auto scopeId = parentId;
      if (apiExternIds.contain(scopeId)) {
        // It's triggered by a CUDA op but not triton op
        scopeId = data->addOp(parentId, kernel->name);
      }
      data->addMetric(scopeId, convertActivityToMetric(activity));
    }
  } else {
    // Graph kernels
    // A single graph launch can trigger multiple kernels.
    // Our solution is to construct the following maps:
    // --- Application threads ---
    // 1. graphId -> numKernels
    // 2. graphExecId -> graphId
    // --- CUPTI thread ---
    // 3. corrId -> numKernels
    for (auto *data : dataSet) {
      auto externId = data->addOp(parentId, kernel->name);
      data->addMetric(externId, convertActivityToMetric(activity));
    }
  }
  apiExternIds.erase(parentId);
  --numInstances;
  if (numInstances == 0) {
    corrIdToExternId.erase(correlationId);
  } else {
    corrIdToExternId[correlationId].second = numInstances;
  }
  return correlationId;
}

uint32_t processActivity(CuptiProfiler::CorrIdToExternIdMap &corrIdToExternId,
                         CuptiProfiler::ApiExternIdSet &apiExternIds,
                         std::set<Data *> &dataSet, CUpti_Activity *activity) {
  auto correlationId = 0;
  switch (activity->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    correlationId = processActivityKernel(corrIdToExternId, apiExternIds,
                                          dataSet, activity);
    break;
  }
  default:
    break;
  }
  return correlationId;
}

// FlagPrism: CUPTI callback IDs are versioned independently from the range
// profiler.  A callback that is valid for ordinary activity collection may be
// rejected with CUPTI_ERROR_INVALID_PARAMETER while a Kernel Replay hardware
// counter session is active.  Keep that optional callback failure from
// aborting the whole NVIDIA vendor session.
bool enableCallbackForSession(CUpti_SubscriberHandle subscriber, bool enable,
                              CUpti_CallbackDomain domain,
                              CUpti_CallbackId callbackId,
                              const char *callbackName,
                              bool tolerateInvalidParameter) {
  const auto result = cupti::enableCallback<false>(
      static_cast<uint32_t>(enable), subscriber, domain, callbackId);
  if (result == CUPTI_SUCCESS) {
    return false;
  }
  if (tolerateInvalidParameter && result == CUPTI_ERROR_INVALID_PARAMETER) {
    return true;
  }
  throw std::runtime_error(
      "Failed to execute cuptiEnableCallback for " + std::string(callbackName) +
      " in domain " + std::to_string(static_cast<uint32_t>(domain)) +
      " with error " + std::to_string(static_cast<uint32_t>(result)));
}

bool setRuntimeCallbacks(CUpti_SubscriberHandle subscriber, bool enable,
                         bool tolerateInvalidParameter) {
  bool skippedCallback = false;
#define CALLBACK_ENABLE(id)                                                    \
  skippedCallback = enableCallbackForSession(subscriber, enable,               \
                                             CUPTI_CB_DOMAIN_RUNTIME_API, id,  \
                                             #id, tolerateInvalidParameter) || \
                    skippedCallback

  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000);
  CALLBACK_ENABLE(
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000);
  CALLBACK_ENABLE(
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000);
  CALLBACK_ENABLE(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);

#undef CALLBACK_ENABLE
  return skippedCallback;
}

bool setDriverCallbacks(CUpti_SubscriberHandle subscriber, bool enable,
                        bool tolerateInvalidParameter) {
  bool skippedCallback = false;
#define CALLBACK_ENABLE(id)                                                    \
  skippedCallback =                                                            \
      enableCallbackForSession(subscriber, enable, CUPTI_CB_DOMAIN_DRIVER_API, \
                               id, #id, tolerateInvalidParameter) ||           \
      skippedCallback

  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunch);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch);
  CALLBACK_ENABLE(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
#undef CALLBACK_ENABLE
  return skippedCallback;
}

bool setGraphCallbacks(CUpti_SubscriberHandle subscriber, bool enable,
                       bool tolerateInvalidParameter) {
  bool skippedCallback = false;

#define CALLBACK_ENABLE(id)                                                    \
  skippedCallback =                                                            \
      enableCallbackForSession(subscriber, enable, CUPTI_CB_DOMAIN_RESOURCE,   \
                               id, #id, tolerateInvalidParameter) ||           \
      skippedCallback

  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING);
#undef CALLBACK_ENABLE
  return skippedCallback;
}

void setResourceCallbacks(CUpti_SubscriberHandle subscriber, bool enable) {
#define CALLBACK_ENABLE(id)                                                    \
  cupti::enableCallback<true>(static_cast<uint32_t>(enable), subscriber,       \
                              CUPTI_CB_DOMAIN_RESOURCE, id)

  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_MODULE_LOADED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
  CALLBACK_ENABLE(CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);
#undef CALLBACK_ENABLE
}

bool isDriverAPILaunch(CUpti_CallbackId cbId) {
  return cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
         cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
}

} // namespace

struct CuptiProfiler::CuptiProfilerPimpl
    : public GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface {
  CuptiProfilerPimpl(CuptiProfiler &profiler)
      : GPUProfiler<CuptiProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~CuptiProfilerPimpl() = default;

  void setLibPath(const std::string &libPath) override {
    cupti::setLibPath(libPath);
    // FlagPrism: Perfworks ships beside the selected CUPTI library in the
    // packaged NVIDIA backend, so reuse the same path for both dispatchers.
    cupti::nvperf::setLibPath(libPath);
    cupti::nvperfTarget::setLibPath(libPath);
  }
  void doStart() override;
  void doFlush() override;
  void doStop() override;

  void setVendorEventCapture(bool enabled, bool captureMemoryActivities) {
    std::lock_guard<std::mutex> lock(vendorMutex);
    vendorEventCapture = enabled;
    vendorMemoryCapture = enabled && captureMemoryActivities;
    // FlagPrism: reset the compact event queue for every new session.  This
    // also prevents events left by an interrupted metric session from being
    // imported by a later base-only session.
    vendorEvents.clear();
    vendorRuntimeDegradeReasons.clear();
    pcSamplingUnavailable = false;
  }

  void setPCSamplingOptional(bool optional) {
    std::lock_guard<std::mutex> lock(vendorMutex);
    pcSamplingOptional = optional;
    pcSamplingUnavailable = false;
  }

  void setHardwareCounterMetrics(const std::vector<std::string> &metricNames) {
    // FlagPrism: configure before start; the helper only allocates NVPW/CUPTI
    // counter images once a session has a current CUDA context.
    hardwareCounters.configure(metricNames);
  }

  void recordVendorRuntimeDegradeReason(const std::string &reason) {
    std::lock_guard<std::mutex> lock(vendorMutex);
    if (std::find(vendorRuntimeDegradeReasons.begin(),
                  vendorRuntimeDegradeReasons.end(),
                  reason) == vendorRuntimeDegradeReasons.end()) {
      vendorRuntimeDegradeReasons.push_back(reason);
    }
  }

  void disableOptionalPCSampling(CUcontext context, const char *phase,
                                 const std::string &reason) {
    // FlagPrism: PC sampling is an optional NVIDIA vendor enhancement. If the
    // driver rejects it (for example CUPTI error 35), keep CUPTI activity
    // profiling alive and report the capability loss in vendor metadata.
    pcSamplingUnavailable = true;
    profiler.disablePCSampling();
    try {
      pcSampling.finalize(context);
    } catch (...) {
      // The cleanup path must not throw from a CUPTI callback.
    }
    try {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
      // FlagPrism: restore the normal concurrent-kernel activity stream after
      // optional PC sampling is rejected, so launch_stats and kernel_duration
      // remain available in the same session.
      cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    } catch (...) {
      // CUPTI may already have torn down the activity stream.
    }
    recordVendorRuntimeDegradeReason(
        "NVIDIA CUPTI PC sampling unavailable during " + std::string(phase) +
        ": " + reason);
  }

  std::vector<RuntimeTraceEventKey> takeVendorRuntimeEvents() {
    std::lock_guard<std::mutex> lock(vendorMutex);
    vendorEventCapture = false;
    // FlagPrism: consume both switches at the session boundary.  Keeping the
    // memory switch alive after export could make a later direct CUPTI start
    // enable memcpy/memset activities before its next vendor plan is applied.
    vendorMemoryCapture = false;
    return std::exchange(vendorEvents, {});
  }

  std::vector<std::string> takeVendorRuntimeDegradeReasons() {
    std::lock_guard<std::mutex> lock(vendorMutex);
    return std::exchange(vendorRuntimeDegradeReasons, {});
  }

  void recordVendorActivity(CUpti_Activity *activity) {
    if (!activity ||
        (activity->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
         activity->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) {
      return;
    }
    auto *kernel = reinterpret_cast<CuptiKernelActivity *>(activity);
    if (kernel->start >= kernel->end) {
      return;
    }

    RuntimeTraceEventKey event;
    event.opName = kernel->name ? kernel->name : "";
    event.correlationId = kernel->correlationId;
    event.deviceId = kernel->deviceId;
    event.streamId = kernel->streamId;
    event.startTimeNs = kernel->start;
    event.endTimeNs = kernel->end;
    event.vendorMetrics["activity_kind"] = std::string("kernel");

    // FlagPrism: CUPTI exposes launch geometry and static resource usage on
    // the activity record itself.  Preserve these properties as typed vendor
    // metrics so `launch_stats` is useful without invoking a second profiler.
    // Dimensions are stored as unsigned values only when CUPTI reports a
    // valid non-negative value; this avoids turning an invalid sentinel into
    // a very large metric.
    const auto addDimension = [&](const char *name, int32_t value) {
      if (value >= 0) {
        event.vendorMetrics[name] = static_cast<uint64_t>(value);
      }
    };
    addDimension("grid_x", kernel->gridX);
    addDimension("grid_y", kernel->gridY);
    addDimension("grid_z", kernel->gridZ);
    addDimension("block_x", kernel->blockX);
    addDimension("block_y", kernel->blockY);
    addDimension("block_z", kernel->blockZ);
    addDimension("static_shared_memory_bytes", kernel->staticSharedMemory);
    addDimension("dynamic_shared_memory_bytes", kernel->dynamicSharedMemory);
    event.vendorMetrics["registers_per_thread"] =
        static_cast<uint64_t>(kernel->registersPerThread);
    event.vendorMetrics["local_memory_per_thread_bytes"] =
        static_cast<uint64_t>(kernel->localMemoryPerThread);
#if CUPTI_API_VERSION >= 17
    // `localMemoryTotal` was deprecated in CUDA 11.8; use the 64-bit field
    // when Kernel9 is available while retaining the legacy fields above.
    event.vendorMetrics["local_memory_total_bytes"] =
        static_cast<uint64_t>(kernel->localMemoryTotal_v2);
    event.vendorMetrics["cluster_x"] = static_cast<uint64_t>(kernel->clusterX);
    event.vendorMetrics["cluster_y"] = static_cast<uint64_t>(kernel->clusterY);
    event.vendorMetrics["cluster_z"] = static_cast<uint64_t>(kernel->clusterZ);
    event.vendorMetrics["max_potential_cluster_size"] =
        static_cast<uint64_t>(kernel->maxPotentialClusterSize);
    event.vendorMetrics["max_active_clusters"] =
        static_cast<uint64_t>(kernel->maxActiveClusters);
#else
    event.vendorMetrics["local_memory_total_bytes"] =
        static_cast<uint64_t>(kernel->localMemoryTotal);
#endif
    event.vendorMetrics["shared_memory_executed_bytes"] =
        static_cast<uint64_t>(kernel->sharedMemoryExecuted);
    event.vendorMetrics["shared_memory_config"] =
        static_cast<uint64_t>(kernel->sharedMemoryConfig);
    event.vendorMetrics["cache_config_requested"] =
        static_cast<uint64_t>(kernel->cacheConfig.config.requested);
    event.vendorMetrics["cache_config_executed"] =
        static_cast<uint64_t>(kernel->cacheConfig.config.executed);
    event.vendorMetrics["launch_type"] =
        static_cast<uint64_t>(kernel->launchType);
    event.vendorMetrics["shared_memory_carveout_requested"] =
        static_cast<uint64_t>(kernel->isSharedMemoryCarveoutRequested);
    event.vendorMetrics["shared_memory_carveout_requested_percent"] =
        static_cast<uint64_t>(kernel->sharedMemoryCarveoutRequested);
    event.vendorMetrics["shmem_limit_config"] =
        static_cast<uint64_t>(kernel->shmemLimitConfig);
    event.vendorMetrics["graph_id"] = static_cast<uint64_t>(kernel->graphId);
    event.vendorMetrics["graph_node_id"] =
        static_cast<uint64_t>(kernel->graphNodeId);
#if CUPTI_API_VERSION >= 17
    event.vendorMetrics["channel_id"] =
        static_cast<uint64_t>(kernel->channelID);
    event.vendorMetrics["channel_type"] =
        static_cast<uint64_t>(kernel->channelType);
#endif
    if (kernel->gridX > 0 && kernel->gridY > 0 && kernel->gridZ > 0) {
      event.vendorMetrics["grid_blocks"] =
          static_cast<uint64_t>(kernel->gridX) *
          static_cast<uint64_t>(kernel->gridY) *
          static_cast<uint64_t>(kernel->gridZ);
    }
    if (kernel->blockX > 0 && kernel->blockY > 0 && kernel->blockZ > 0) {
      event.vendorMetrics["block_threads"] =
          static_cast<uint64_t>(kernel->blockX) *
          static_cast<uint64_t>(kernel->blockY) *
          static_cast<uint64_t>(kernel->blockZ);
    }
    if (profiler.correlation.corrIdToExternId.contain(kernel->correlationId)) {
      const auto parentId =
          profiler.correlation.corrIdToExternId.at(kernel->correlationId).first;
      // FlagPrism: graph and external CUDA kernels receive child scopes while
      // base CUPTI activities are imported. The vendor callback runs before
      // that import, so attaching to parentId would associate the vendor event
      // with the wrong node. Keep the event as a synthetic, time-based scope
      // in those cases; ordinary Triton kernels can reuse the parent scope.
      if (kernel->graphId == 0 &&
          !profiler.correlation.apiExternIds.contain(parentId)) {
        event.scopeId = parentId;
      }
    }

    // FlagPrism: AutoRange indexes hardware-counter ranges by kernel launch
    // order. Record the same compact event independently of the vendor queue
    // so replay metadata can be joined after CUPTI has flushed its buffers.
    hardwareCounters.recordKernel(event);

    std::lock_guard<std::mutex> lock(vendorMutex);
    if (vendorEventCapture) {
      // FlagPrism: retain a compact, backend-neutral event key for the
      // NVIDIA importer; CUPTI buffers themselves are released below.  The
      // optional launch properties above remain in the same key so the
      // importer can correlate them without a second pass over CUPTI data.
      vendorEvents.push_back(std::move(event));
    }
  }

  void recordVendorMemoryActivity(CUpti_Activity *activity) {
    if (!activity || (activity->kind != CUPTI_ACTIVITY_KIND_MEMCPY &&
                      activity->kind != CUPTI_ACTIVITY_KIND_MEMSET)) {
      return;
    }

    RuntimeTraceEventKey event;
    uint32_t correlationId = 0;
    uint32_t graphId = 0;
    uint64_t bytes = 0;
    if (activity->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto *memory = reinterpret_cast<CuptiMemcpyActivity *>(activity);
      if (memory->start >= memory->end) {
        return;
      }
      event.opName = "cuda_memcpy";
      event.startTimeNs = memory->start;
      event.endTimeNs = memory->end;
      event.deviceId = memory->deviceId;
      event.streamId = memory->streamId;
      correlationId = memory->correlationId;
      graphId = memory->graphId;
      bytes = memory->bytes;
      event.vendorMetrics["activity_kind"] = std::string("memcpy");
      event.vendorMetrics["memory_bytes"] = bytes;
      event.vendorMetrics["memory_copy_kind"] =
          static_cast<uint64_t>(memory->copyKind);
      event.vendorMetrics["memory_source_kind"] =
          static_cast<uint64_t>(memory->srcKind);
      event.vendorMetrics["memory_destination_kind"] =
          static_cast<uint64_t>(memory->dstKind);
#if CUPTI_API_VERSION >= 26
      event.vendorMetrics["memory_copy_count"] =
          static_cast<uint64_t>(memory->copyCount);
#else
      // FlagPrism: Memcpy5 has no copyCount field; each activity record still
      // represents one completed transfer.
      event.vendorMetrics["memory_copy_count"] = static_cast<uint64_t>(1);
#endif
    } else {
      auto *memory = reinterpret_cast<CuptiMemsetActivity *>(activity);
      if (memory->start >= memory->end) {
        return;
      }
      event.opName = "cuda_memset";
      event.startTimeNs = memory->start;
      event.endTimeNs = memory->end;
      event.deviceId = memory->deviceId;
      event.streamId = memory->streamId;
      correlationId = memory->correlationId;
      graphId = memory->graphId;
      bytes = memory->bytes;
      event.vendorMetrics["activity_kind"] = std::string("memset");
      event.vendorMetrics["memory_bytes"] = bytes;
      event.vendorMetrics["memory_set_value"] =
          static_cast<uint64_t>(memory->value);
      event.vendorMetrics["memory_kind"] =
          static_cast<uint64_t>(memory->memoryKind);
    }
    event.correlationId = correlationId;
    const auto durationNs = event.endTimeNs - event.startTimeNs;
    if (durationNs > 0) {
      // FlagPrism: bytes/ns is numerically equal to decimal GB/s.  This is a
      // transfer-rate estimate from activity timestamps, not a DRAM counter.
      event.vendorMetrics["memory_bandwidth_gb_s"] =
          static_cast<double>(bytes) / static_cast<double>(durationNs);
    }

    if (profiler.correlation.corrIdToExternId.contain(correlationId)) {
      const auto parentId =
          profiler.correlation.corrIdToExternId.at(correlationId).first;
      if (graphId == 0 &&
          !profiler.correlation.apiExternIds.contain(parentId)) {
        event.scopeId = parentId;
      }
    }

    std::lock_guard<std::mutex> lock(vendorMutex);
    if (vendorEventCapture) {
      // FlagPrism: memory activities share the same queue and correlation
      // schema as kernel activities, so the common artifact overlay can
      // associate both without a second CUDA collector.
      vendorEvents.push_back(std::move(event));
    }
  }

  void recordVendorPCSampling(size_t scopeId, const std::string &opName,
                              const std::string &stallMetricName,
                              uint64_t samples, uint64_t stalledSamples) {
    if (samples == 0) {
      return;
    }

    RuntimeTraceEventKey event;
    event.scopeId = scopeId;
    event.opName = opName;
    event.vendorMetrics["activity_kind"] = std::string("pcsampling");
    event.vendorMetrics["instruction_samples"] = samples;
    event.vendorMetrics["instruction_stalled_samples"] = stalledSamples;
    // FlagPrism: retain each CUPTI stall-reason bucket with a stable vendor
    // prefix. These are sampled instruction observations, not PM counter
    // values, and can be overlaid without replacing the legacy PC metric.
    // FlagPrism: stall-reason buckets use stalledSamples, which preserves the
    // not-issued bucket semantics instead of duplicating total samples.
    event.vendorMetrics["instruction_" + stallMetricName] = stalledSamples;

    std::lock_guard<std::mutex> lock(vendorMutex);
    if (vendorEventCapture) {
      vendorEvents.push_back(std::move(event));
    }
  }

  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords);
  static void completeBuffer(CUcontext context, uint32_t streamId,
                             uint8_t *buffer, size_t size, size_t validSize);
  static void callbackFn(void *userData, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbId, const void *cbData);

  static constexpr size_t AlignSize = 8;
  static constexpr size_t BufferSize = 64 * 1024 * 1024;
  static constexpr size_t AttributeSize = sizeof(size_t);

  CUpti_SubscriberHandle subscriber{};
  CuptiPCSampling pcSampling;
  NvidiaHardwareCounters hardwareCounters;

  ThreadSafeMap<uint32_t, size_t, std::unordered_map<uint32_t, size_t>>
      graphIdToNumInstances;
  ThreadSafeMap<uint32_t, uint32_t, std::unordered_map<uint32_t, uint32_t>>
      graphExecIdToGraphId;
  std::mutex vendorMutex;
  bool vendorEventCapture = false;
  bool vendorMemoryCapture = false;
  std::atomic<bool> pcSamplingOptional{false};
  bool vendorMemoryActivitiesEnabled = false;
  std::vector<RuntimeTraceEventKey> vendorEvents;
  std::vector<std::string> vendorRuntimeDegradeReasons;
  std::atomic<bool> pcSamplingUnavailable{false};
};

void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
  if (*buffer == nullptr) {
    throw std::runtime_error("[PROTON] aligned_alloc failed");
  }
  *bufferSize = BufferSize;
  *maxNumRecords = 0;
}

void CuptiProfiler::CuptiProfilerPimpl::completeBuffer(CUcontext ctx,
                                                       uint32_t streamId,
                                                       uint8_t *buffer,
                                                       size_t size,
                                                       size_t validSize) {
  CuptiProfiler &profiler = threadState.profiler;
  auto dataSet = profiler.getDataSet();
  uint32_t maxCorrelationId = 0;
  CUptiResult status;
  CUpti_Activity *activity = nullptr;
  do {
    status = cupti::activityGetNextRecord<false>(buffer, validSize, &activity);
    if (status == CUPTI_SUCCESS) {
      // FlagPrism: collect the event before processActivity consumes the
      // correlation map entry for the final kernel in a launch.
      auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
      pImpl->recordVendorActivity(activity);
      pImpl->recordVendorMemoryActivity(activity);
      auto correlationId =
          processActivity(profiler.correlation.corrIdToExternId,
                          profiler.correlation.apiExternIds, dataSet, activity);
      maxCorrelationId = std::max(maxCorrelationId, correlationId);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      throw std::runtime_error("[PROTON] cupti::activityGetNextRecord failed");
    }
  } while (true);

  std::free(buffer);

  profiler.correlation.complete(maxCorrelationId);
}

void CuptiProfiler::CuptiProfilerPimpl::callbackFn(void *userData,
                                                   CUpti_CallbackDomain domain,
                                                   CUpti_CallbackId cbId,
                                                   const void *cbData) {
  CuptiProfiler &profiler = threadState.profiler;
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    auto *resourceData =
        static_cast<CUpti_ResourceData *>(const_cast<void *>(cbData));
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (cbId == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.loadModule(moduleResource->pCubin,
                                     moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
      auto *moduleResource = static_cast<CUpti_ModuleResourceData *>(
          resourceData->resourceDescriptor);
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.unloadModule(moduleResource->pCubin,
                                       moduleResource->cubinSize);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.initialize(resourceData->context);
      }
    } else if (cbId == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      if (profiler.isPCSamplingEnabled()) {
        pImpl->pcSampling.finalize(resourceData->context);
      }
    } else {
      auto *graphData =
          static_cast<CUpti_GraphData *>(resourceData->resourceDescriptor);
      uint32_t graphId = 0;
      uint32_t graphExecId = 0;
      if (graphData->graph)
        cupti::getGraphId<true>(graphData->graph, &graphId);
      if (graphData->graphExec)
        cupti::getGraphExecId<true>(graphData->graphExec, &graphExecId);
      if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED ||
          cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
        if (!pImpl->graphIdToNumInstances.contain(graphId))
          pImpl->graphIdToNumInstances[graphId] = 1;
        else
          pImpl->graphIdToNumInstances[graphId]++;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING) {
        pImpl->graphIdToNumInstances[graphId]--;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED) {
        pImpl->graphExecIdToGraphId[graphExecId] = graphId;
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING) {
        pImpl->graphExecIdToGraphId.erase(graphExecId);
      } else if (cbId == CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING) {
        pImpl->graphIdToNumInstances.erase(graphId);
      }
    }
  } else {
    const CUpti_CallbackData *callbackData =
        static_cast<const CUpti_CallbackData *>(cbData);
    auto *pImpl = dynamic_cast<CuptiProfilerPimpl *>(profiler.pImpl.get());
    if (callbackData->callbackSite == CUPTI_API_ENTER) {
      threadState.enterOp();
      size_t numInstances = 1;
      if (cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
          cbId == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
        auto graphExec = static_cast<const cuGraphLaunch_params *>(
                             callbackData->functionParams)
                             ->hGraph;
        uint32_t graphExecId = 0;
        cupti::getGraphExecId<true>(graphExec, &graphExecId);
        numInstances = std::numeric_limits<size_t>::max();
        auto findGraph = false;
        if (pImpl->graphExecIdToGraphId.contain(graphExecId)) {
          auto graphId = pImpl->graphExecIdToGraphId[graphExecId];
          if (pImpl->graphIdToNumInstances.contain(graphId)) {
            numInstances = pImpl->graphIdToNumInstances[graphId];
            findGraph = true;
          }
        }
        if (!findGraph)
          std::cerr << "[PROTON] Cannot find graph for graphExecId: "
                    << graphExecId
                    << ", and t may cause memory leak. To avoid this problem, "
                       "please start profiling before the graph is created."
                    << std::endl;
      }
      profiler.correlation.correlate(callbackData->correlationId, numInstances);
      if (profiler.isPCSamplingEnabled() && isDriverAPILaunch(cbId)) {
        try {
          pImpl->pcSampling.start(callbackData->context);
        } catch (const std::exception &error) {
          if (!pImpl->pcSamplingOptional)
            throw;
          pImpl->disableOptionalPCSampling(callbackData->context, "start",
                                           error.what());
        }
      }
    } else if (callbackData->callbackSite == CUPTI_API_EXIT) {
      if (profiler.isPCSamplingEnabled() && isDriverAPILaunch(cbId)) {
        // XXX: Conservatively stop every GPU kernel for now
        auto scopeId = profiler.correlation.externIdQueue.back();
        try {
          pImpl->pcSampling.stop(
              callbackData->context, scopeId,
              profiler.correlation.apiExternIds.contain(scopeId));
        } catch (const std::exception &error) {
          if (!pImpl->pcSamplingOptional)
            throw;
          pImpl->disableOptionalPCSampling(callbackData->context, "stop",
                                           error.what());
        }
      }
      threadState.exitOp();
      profiler.correlation.submit(callbackData->correlationId);
    }
  }
}

void CuptiProfiler::CuptiProfilerPimpl::doStart() {
  cupti::subscribe<true>(&subscriber, callbackFn, nullptr);
  // FlagPrism: hardware-counter sessions use CUPTI Kernel Replay.  Keep
  // callback registration best-effort for this mode because CUPTI can reject
  // individual callback IDs without making activity collection unusable.
  const bool tolerateUnsupportedCallbacks = hardwareCounters.isConfigured();
  if (hardwareCounters.isConfigured()) {
    CUcontext cuContext = nullptr;
    cuda::ctxGetCurrent<false>(&cuContext);
    std::vector<std::string> hardwareDegradeReasons;
    hardwareCounters.start(cuContext, hardwareDegradeReasons);
    for (const auto &reason : hardwareDegradeReasons) {
      recordVendorRuntimeDegradeReason(reason);
    }
  }
  // FlagPrism: collect memcpy/memset activities whenever the NVIDIA vendor
  // stream is enabled.  The importer filters them unless `memory` was
  // requested, while keeping a single CUPTI subscription for all metrics.
  vendorMemoryActivitiesEnabled = vendorEventCapture && vendorMemoryCapture;
  if (vendorMemoryActivitiesEnabled) {
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_MEMCPY);
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_MEMSET);
  }
  if (profiler.isPCSamplingEnabled()) {
    setResourceCallbacks(subscriber, /*enable=*/true);
    // Continuous PC sampling is not compatible with concurrent kernel profiling
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
  } else {
    cupti::activityEnable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }
  cupti::activityRegisterCallbacks<true>(allocBuffer, completeBuffer);
  const bool skippedGraphCallbacks = setGraphCallbacks(
      subscriber, /*enable=*/true, tolerateUnsupportedCallbacks);
  const bool skippedRuntimeCallbacks = setRuntimeCallbacks(
      subscriber, /*enable=*/true, tolerateUnsupportedCallbacks);
  const bool skippedDriverCallbacks = setDriverCallbacks(
      subscriber, /*enable=*/true, tolerateUnsupportedCallbacks);
  if (skippedGraphCallbacks || skippedRuntimeCallbacks ||
      skippedDriverCallbacks) {
    recordVendorRuntimeDegradeReason(
        "NVIDIA CUPTI skipped callback IDs that are invalid during the "
        "hardware-counter session.");
  }
}

void CuptiProfiler::CuptiProfilerPimpl::doFlush() {
  // cuptiActivityFlushAll returns the activity records associated with all
  // contexts/streams.
  // This is a blocking call but it doesn’t issue any CUDA synchronization calls
  // implicitly thus it’s not guaranteed that all activities are completed on
  // the underlying devices.
  // We do an "opportunistic" synchronization here to try to ensure that all
  // activities are completed on the current context.
  // If the current context is not set, we don't do any synchronization.
  CUcontext cuContext = nullptr;
  cuda::ctxGetCurrent<false>(&cuContext);
  if (cuContext) {
    cuda::ctxSynchronize<true>();
  }
  try {
    profiler.correlation.flush(
        /*maxRetries=*/100, /*sleepMs=*/10,
        /*flush=*/[]() {
          cupti::activityFlushAll<true>(
              /*flag=*/0);
        });
    // CUPTI_ACTIVITY_FLAG_FLUSH_FORCED is used to ensure that even incomplete
    // activities are flushed so that the next profiling session can start with
    // new activities.
    cupti::activityFlushAll<true>(/*flag=*/CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  } catch (const std::exception &error) {
    if (!pcSamplingOptional || !pcSamplingUnavailable)
      throw;
    // FlagPrism: some drivers leave CUPTI activity flushing in an unavailable
    // state after rejecting PC sampling. The vendor artifact already records
    // that loss; keep finalization non-fatal for the optional path.
    recordVendorRuntimeDegradeReason(
        "NVIDIA CUPTI activity flush unavailable after PC sampling fallback: " +
        std::string(error.what()));
  }
}

void CuptiProfiler::CuptiProfilerPimpl::doStop() {
  // FlagPrism: finalize the optional NVPW session while the CUPTI activity
  // subscription is still alive. KernelReplay counter data is flushed here;
  // the resulting events are then consumed by the NVIDIA importer together
  // with the ordinary CUPTI activity events.
  if (hardwareCounters.isActive()) {
    std::vector<RuntimeTraceEventKey> hardwareEvents;
    std::vector<std::string> hardwareDegradeReasons;
    hardwareCounters.stop(hardwareEvents, hardwareDegradeReasons);
    {
      std::lock_guard<std::mutex> lock(vendorMutex);
      vendorEvents.insert(vendorEvents.end(),
                          std::make_move_iterator(hardwareEvents.begin()),
                          std::make_move_iterator(hardwareEvents.end()));
    }
    for (const auto &reason : hardwareDegradeReasons) {
      recordVendorRuntimeDegradeReason(reason);
    }
  }
  const bool tolerateCleanupErrors =
      pcSamplingOptional && pcSamplingUnavailable;
  const bool tolerateUnsupportedCallbacks = hardwareCounters.isConfigured();
  auto runCleanup = [&](const char *phase, const auto &cleanup) {
    try {
      cleanup();
    } catch (const std::exception &error) {
      if (!tolerateCleanupErrors)
        throw;
      // FlagPrism: after a driver-side PC sampling rejection, CUPTI may return
      // the same capability error from teardown calls. Keep finalization
      // non-fatal and report each failed cleanup operation.
      recordVendorRuntimeDegradeReason(
          "NVIDIA CUPTI cleanup unavailable during " + std::string(phase) +
          ": " + error.what());
    }
  };

  if (profiler.isPCSamplingEnabled()) {
    profiler.disablePCSampling();
    CUcontext cuContext = nullptr;
    cuda::ctxGetCurrent<false>(&cuContext);
    if (cuContext)
      runCleanup("PC sampling finalize",
                 [&]() { pcSampling.finalize(cuContext); });
    runCleanup("resource callback disable",
               [&]() { setResourceCallbacks(subscriber, /*enable=*/false); });
    runCleanup("kernel activity disable", [&]() {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_KERNEL);
    });
  } else {
    runCleanup("concurrent-kernel activity disable", [&]() {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    });
  }
  if (vendorMemoryActivitiesEnabled) {
    runCleanup("memcpy activity disable", [&]() {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_MEMCPY);
    });
    runCleanup("memset activity disable", [&]() {
      cupti::activityDisable<true>(CUPTI_ACTIVITY_KIND_MEMSET);
    });
    vendorMemoryActivitiesEnabled = false;
  }
  runCleanup("graph callback disable", [&]() {
    setGraphCallbacks(subscriber, /*enable=*/false,
                      tolerateUnsupportedCallbacks);
  });
  runCleanup("runtime callback disable", [&]() {
    setRuntimeCallbacks(subscriber, /*enable=*/false,
                        tolerateUnsupportedCallbacks);
  });
  runCleanup("driver callback disable", [&]() {
    setDriverCallbacks(subscriber, /*enable=*/false,
                       tolerateUnsupportedCallbacks);
  });
  runCleanup("CUPTI unsubscribe",
             [&]() { cupti::unsubscribe<true>(subscriber); });
  runCleanup("CUPTI finalize", [&]() { cupti::finalize<true>(); });
}

CuptiProfiler::CuptiProfiler() {
  pImpl = std::make_unique<CuptiProfilerPimpl>(*this);
}

CuptiProfiler::~CuptiProfiler() = default;

void CuptiProfiler::enableVendorEventCapture(bool enabled,
                                             bool captureMemoryActivities) {
  auto *implementation = dynamic_cast<CuptiProfilerPimpl *>(this->pImpl.get());
  implementation->setVendorEventCapture(enabled, captureMemoryActivities);
}

void CuptiProfiler::setPCSamplingOptional(bool optional) {
  auto *implementation = dynamic_cast<CuptiProfilerPimpl *>(this->pImpl.get());
  implementation->setPCSamplingOptional(optional);
}

void CuptiProfiler::setHardwareCounterMetrics(
    const std::vector<std::string> &metricNames) {
  auto *implementation = dynamic_cast<CuptiProfilerPimpl *>(this->pImpl.get());
  implementation->setHardwareCounterMetrics(metricNames);
}

void CuptiProfiler::recordVendorPCSampling(size_t scopeId,
                                           const std::string &opName,
                                           const std::string &stallMetricName,
                                           uint64_t samples,
                                           uint64_t stalledSamples) {
  auto *implementation = dynamic_cast<CuptiProfilerPimpl *>(this->pImpl.get());
  implementation->recordVendorPCSampling(scopeId, opName, stallMetricName,
                                         samples, stalledSamples);
}

std::vector<std::string> CuptiProfiler::takeVendorRuntimeDegradeReasons() {
  auto *implementation = dynamic_cast<CuptiProfilerPimpl *>(this->pImpl.get());
  return implementation->takeVendorRuntimeDegradeReasons();
}

std::vector<RuntimeTraceEventKey> CuptiProfiler::takeVendorRuntimeEvents() {
  auto *implementation = dynamic_cast<CuptiProfilerPimpl *>(this->pImpl.get());
  return implementation->takeVendorRuntimeEvents();
}

} // namespace proton
