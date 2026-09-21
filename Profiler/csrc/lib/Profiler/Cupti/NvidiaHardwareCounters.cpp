#include "Profiler/Cupti/NvidiaHardwareCounters.h"

#include <algorithm>
#include <exception>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace proton {
namespace nvperf = cupti::nvperf;
namespace nvperfTarget = cupti::nvperfTarget;
namespace {

constexpr const char *kOccupancyMetric =
    "sm__warps_active.avg.pct_of_peak_sustained_active";
constexpr const char *kThroughputMetric =
    "sm__throughput.avg.pct_of_peak_sustained_elapsed";
constexpr const char *kBandwidthMetric =
    "dram__throughput.avg.pct_of_peak_sustained_elapsed";
constexpr const char *kInstructionCountMetric = "smsp__inst_executed.sum";

std::string exceptionMessage(const std::exception &error) {
  return error.what();
}

} // namespace

NvidiaHardwareCounters::~NvidiaHardwareCounters() {
  // FlagPrism: normal sessions call stop() explicitly.  Keep the destructor
  // as a last-resort cleanup guard for an exception during session setup, but
  // do not attempt evaluation while the Python/CUDA runtime may be shutting
  // down.
  if (profilerInitialized_ || sessionBegun_ || configSet_ ||
      profilingEnabled_) {
    cleanup(nullptr);
  }
}

void NvidiaHardwareCounters::configure(
    const std::vector<std::string> &metricNames, size_t maxRanges) {
  std::lock_guard<std::mutex> lock(mutex_);
  configured_ = false;
  active_ = false;
  metrics_.clear();
  kernelEvents_.clear();
  maxRanges_ = std::max<size_t>(1, maxRanges);

  for (const auto &name : metricNames) {
    if (name == "occupancy") {
      metrics_.push_back({name, kOccupancyMetric});
    } else if (name == "throughput") {
      metrics_.push_back({name, kThroughputMetric});
    } else if (name == "bandwidth") {
      metrics_.push_back({name, kBandwidthMetric});
    } else if (name == "instruction_count") {
      metrics_.push_back({name, kInstructionCountMetric});
    }
  }
  configured_ = !metrics_.empty();
}

bool NvidiaHardwareCounters::isConfigured() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return configured_;
}

bool NvidiaHardwareCounters::isActive() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

void NvidiaHardwareCounters::appendDegrade(
    std::vector<std::string> &degradeReasons, const std::string &reason) {
  if (std::find(degradeReasons.begin(), degradeReasons.end(), reason) ==
      degradeReasons.end()) {
    degradeReasons.push_back(reason);
  }
}

void NvidiaHardwareCounters::start(CUcontext context,
                                   std::vector<std::string> &degradeReasons) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = false;
    kernelEvents_.clear();
  }
  if (!isConfigured()) {
    return;
  }
  if (context == nullptr) {
    appendDegrade(degradeReasons,
                  "NVIDIA NVPW hardware-counter profiling requires a current "
                  "CUDA "
                  "context.");
    return;
  }

  try {
    context_ = context;

    // FlagPrism: CUPTI owns the device-side profiler session while NVPW
    // creates the raw-counter configuration and counter-data prefix.
    CUpti_Profiler_Initialize_Params initializeParams = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    cupti::profilerInitialize<true>(&initializeParams);
    profilerInitialized_ = true;

    CUdevice device{};
    cuda::ctxGetDevice<true>(&device);
    CUpti_Device_GetChipName_Params chipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipNameParams.deviceIndex = static_cast<size_t>(device);
    cupti::deviceGetChipName<true>(&chipNameParams);
    if (!chipNameParams.pChipName || chipNameParams.pChipName[0] == '\0') {
      throw std::runtime_error("CUPTI returned an empty NVIDIA chip name");
    }
    chipName_ = chipNameParams.pChipName;

    CUpti_Profiler_GetCounterAvailability_Params availabilityParams = {
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    availabilityParams.ctx = context_;
    cupti::profilerGetCounterAvailability<true>(&availabilityParams);
    if (availabilityParams.counterAvailabilityImageSize == 0) {
      throw std::runtime_error(
          "CUPTI returned an empty counter-availability image");
    }
    counterAvailability_.resize(
        availabilityParams.counterAvailabilityImageSize);
    availabilityParams.pCounterAvailabilityImage = counterAvailability_.data();
    cupti::profilerGetCounterAvailability<true>(&availabilityParams);

    NVPW_InitializeHost_Params initializeHostParams = {
        NVPW_InitializeHost_Params_STRUCT_SIZE};
    nvperf::initializeHost<true>(&initializeHostParams);

    // Build the raw metric dependency list from the derived NVPW metric.  The
    // dependency names are copied before the evaluator is destroyed because
    // NVPW owns the returned strings.
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params evaluatorSize = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    evaluatorSize.pChipName = chipName_.c_str();
    evaluatorSize.pCounterAvailabilityImage = counterAvailability_.data();
    nvperf::cudaMetricsEvaluatorCalculateScratchBufferSize<true>(
        &evaluatorSize);
    std::vector<uint8_t> evaluatorScratch(evaluatorSize.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params evaluatorInitialize = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    evaluatorInitialize.pScratchBuffer = evaluatorScratch.data();
    evaluatorInitialize.scratchBufferSize = evaluatorScratch.size();
    evaluatorInitialize.pChipName = chipName_.c_str();
    evaluatorInitialize.pCounterAvailabilityImage = counterAvailability_.data();
    nvperf::cudaMetricsEvaluatorInitialize<true>(&evaluatorInitialize);
    auto *metricsEvaluator = evaluatorInitialize.pMetricsEvaluator;
    if (!metricsEvaluator) {
      throw std::runtime_error("NVPW returned a null metrics evaluator");
    }

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<NVPW_MetricEvalRequest> metricEvalRequests;
    std::vector<std::string> rawMetricNames;
    try {
      metricEvalRequests.reserve(metrics_.size());
      for (const auto &metric : metrics_) {
        NVPW_MetricEvalRequest metricEvalRequest{};
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
            convertParams = {
                NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        convertParams.pMetricsEvaluator = metricsEvaluator;
        convertParams.pMetricName = metric.evaluatorName.c_str();
        convertParams.pMetricEvalRequest = &metricEvalRequest;
        convertParams.metricEvalRequestStructSize =
            NVPW_MetricEvalRequest_STRUCT_SIZE;
        nvperf::metricsEvaluatorConvertMetricNameToMetricEvalRequest<true>(
            &convertParams);
        metricEvalRequests.push_back(metricEvalRequest);

        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params dependencyParams =
            {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
        dependencyParams.pMetricsEvaluator = metricsEvaluator;
        dependencyParams.pMetricEvalRequests = &metricEvalRequests.back();
        dependencyParams.numMetricEvalRequests = 1;
        dependencyParams.metricEvalRequestStructSize =
            NVPW_MetricEvalRequest_STRUCT_SIZE;
        dependencyParams.metricEvalRequestStrideSize =
            sizeof(NVPW_MetricEvalRequest);
        nvperf::metricsEvaluatorGetMetricRawDependencies<true>(
            &dependencyParams);
        std::vector<const char *> dependencies(
            dependencyParams.numRawDependencies);
        dependencyParams.ppRawDependencies = dependencies.data();
        nvperf::metricsEvaluatorGetMetricRawDependencies<true>(
            &dependencyParams);
        for (const auto *dependency : dependencies) {
          if (dependency &&
              std::find(rawMetricNames.begin(), rawMetricNames.end(),
                        dependency) == rawMetricNames.end()) {
            rawMetricNames.emplace_back(dependency);
          }
        }
      }

      rawMetricRequests.reserve(rawMetricNames.size());
      for (const auto &rawMetricName : rawMetricNames) {
        NVPA_RawMetricRequest request = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
        request.pMetricName = rawMetricName.c_str();
        request.isolated = 1;
        request.keepInstances = 1;
        rawMetricRequests.push_back(request);
      }

      NVPW_CUDA_RawMetricsConfig_Create_V2_Params configCreate = {
          NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
      configCreate.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
      configCreate.pChipName = chipName_.c_str();
      configCreate.pCounterAvailabilityImage = counterAvailability_.data();
      nvperf::cudaRawMetricsConfigCreateV2<true>(&configCreate);
      auto *rawConfig = configCreate.pRawMetricsConfig;
      if (!rawConfig) {
        throw std::runtime_error("NVPW returned a null raw metrics config");
      }
      try {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setAvailability = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
        setAvailability.pRawMetricsConfig = rawConfig;
        setAvailability.pCounterAvailabilityImage = counterAvailability_.data();
        nvperf::rawMetricsConfigSetCounterAvailability<true>(&setAvailability);

        NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroup = {
            NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
        beginPassGroup.pRawMetricsConfig = rawConfig;
        nvperf::rawMetricsConfigBeginPassGroup<true>(&beginPassGroup);
        NVPW_RawMetricsConfig_AddMetrics_Params addMetrics = {
            NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
        addMetrics.pRawMetricsConfig = rawConfig;
        addMetrics.pRawMetricRequests = rawMetricRequests.data();
        addMetrics.numMetricRequests = rawMetricRequests.size();
        nvperf::rawMetricsConfigAddMetrics<true>(&addMetrics);
        NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroup = {
            NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
        endPassGroup.pRawMetricsConfig = rawConfig;
        nvperf::rawMetricsConfigEndPassGroup<true>(&endPassGroup);
        NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfig = {
            NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
        generateConfig.pRawMetricsConfig = rawConfig;
        generateConfig.mergeAllPassGroups = 0;
        nvperf::rawMetricsConfigGenerateConfigImage<true>(&generateConfig);
        NVPW_RawMetricsConfig_GetConfigImage_Params getConfig = {
            NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
        getConfig.pRawMetricsConfig = rawConfig;
        nvperf::rawMetricsConfigGetConfigImage<true>(&getConfig);
        if (getConfig.bytesCopied == 0) {
          throw std::runtime_error("NVPW generated an empty config image");
        }
        configImage_.resize(getConfig.bytesCopied);
        getConfig.bytesAllocated = configImage_.size();
        getConfig.pBuffer = configImage_.data();
        nvperf::rawMetricsConfigGetConfigImage<true>(&getConfig);
      } catch (...) {
        NVPW_RawMetricsConfig_Destroy_Params destroyConfig = {
            NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
        destroyConfig.pRawMetricsConfig = rawConfig;
        nvperf::rawMetricsConfigDestroy<false>(&destroyConfig);
        throw;
      }
      NVPW_RawMetricsConfig_Destroy_Params destroyConfig = {
          NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
      destroyConfig.pRawMetricsConfig = rawConfig;
      nvperf::rawMetricsConfigDestroy<true>(&destroyConfig);

      NVPW_CUDA_CounterDataBuilder_Create_Params builderCreate = {
          NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE};
      builderCreate.pChipName = chipName_.c_str();
      builderCreate.pCounterAvailabilityImage = counterAvailability_.data();
      nvperf::cudaCounterDataBuilderCreate<true>(&builderCreate);
      auto *counterDataBuilder = builderCreate.pCounterDataBuilder;
      if (!counterDataBuilder) {
        throw std::runtime_error("NVPW returned a null counter-data builder");
      }
      try {
        NVPW_CounterDataBuilder_AddMetrics_Params addMetrics = {
            NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
        addMetrics.pCounterDataBuilder = counterDataBuilder;
        addMetrics.pRawMetricRequests = rawMetricRequests.data();
        addMetrics.numMetricRequests = rawMetricRequests.size();
        nvperf::counterDataBuilderAddMetrics<true>(&addMetrics);
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getPrefix = {
            NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
        getPrefix.pCounterDataBuilder = counterDataBuilder;
        nvperf::counterDataBuilderGetCounterDataPrefix<true>(&getPrefix);
        if (getPrefix.bytesCopied == 0) {
          throw std::runtime_error(
              "NVPW generated an empty counter-data prefix");
        }
        counterDataPrefix_.resize(getPrefix.bytesCopied);
        getPrefix.bytesAllocated = counterDataPrefix_.size();
        getPrefix.pBuffer = counterDataPrefix_.data();
        nvperf::counterDataBuilderGetCounterDataPrefix<true>(&getPrefix);
      } catch (...) {
        NVPW_CounterDataBuilder_Destroy_Params destroyBuilder = {
            NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
        destroyBuilder.pCounterDataBuilder = counterDataBuilder;
        nvperf::counterDataBuilderDestroy<false>(&destroyBuilder);
        throw;
      }
      NVPW_CounterDataBuilder_Destroy_Params destroyBuilder = {
          NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
      destroyBuilder.pCounterDataBuilder = counterDataBuilder;
      nvperf::counterDataBuilderDestroy<true>(&destroyBuilder);
    } catch (...) {
      NVPW_MetricsEvaluator_Destroy_Params destroyEvaluator = {
          NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
      destroyEvaluator.pMetricsEvaluator = metricsEvaluator;
      nvperf::metricsEvaluatorDestroy<false>(&destroyEvaluator);
      throw;
    }
    NVPW_MetricsEvaluator_Destroy_Params destroyEvaluator = {
        NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    destroyEvaluator.pMetricsEvaluator = metricsEvaluator;
    nvperf::metricsEvaluatorDestroy<true>(&destroyEvaluator);

    CUpti_Profiler_CounterDataImageOptions imageOptions = {
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE};
    imageOptions.pCounterDataPrefix = counterDataPrefix_.data();
    imageOptions.counterDataPrefixSize = counterDataPrefix_.size();
    imageOptions.maxNumRanges = static_cast<uint32_t>(maxRanges_);
    imageOptions.maxNumRangeTreeNodes = static_cast<uint32_t>(maxRanges_);
    imageOptions.maxRangeNameLength = 128;
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateImage = {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
    calculateImage.sizeofCounterDataImageOptions =
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    calculateImage.pOptions = &imageOptions;
    cupti::profilerCounterDataImageCalculateSize<true>(&calculateImage);
    if (calculateImage.counterDataImageSize == 0) {
      throw std::runtime_error("CUPTI calculated an empty counter-data image");
    }
    counterDataImage_.resize(calculateImage.counterDataImageSize);
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeImage = {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeImage.sizeofCounterDataImageOptions =
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeImage.pOptions = &imageOptions;
    initializeImage.counterDataImageSize = counterDataImage_.size();
    initializeImage.pCounterDataImage = counterDataImage_.data();
    cupti::profilerCounterDataImageInitialize<true>(&initializeImage);

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
        calculateScratch = {
            CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratch.counterDataImageSize = counterDataImage_.size();
    calculateScratch.pCounterDataImage = counterDataImage_.data();
    cupti::profilerCounterDataImageCalculateScratchBufferSize<true>(
        &calculateScratch);
    counterDataScratchBuffer_.resize(
        calculateScratch.counterDataScratchBufferSize);
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
        initializeScratch = {
            CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initializeScratch.counterDataImageSize = counterDataImage_.size();
    initializeScratch.pCounterDataImage = counterDataImage_.data();
    initializeScratch.counterDataScratchBufferSize =
        counterDataScratchBuffer_.size();
    initializeScratch.pCounterDataScratchBuffer =
        counterDataScratchBuffer_.data();
    cupti::profilerCounterDataImageInitializeScratchBuffer<true>(
        &initializeScratch);

    CUpti_Profiler_BeginSession_Params beginSession = {
        CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSession.ctx = context_;
    beginSession.counterDataImageSize = counterDataImage_.size();
    beginSession.pCounterDataImage = counterDataImage_.data();
    beginSession.counterDataScratchBufferSize =
        counterDataScratchBuffer_.size();
    beginSession.pCounterDataScratchBuffer = counterDataScratchBuffer_.data();
    beginSession.range = CUPTI_AutoRange;
    beginSession.replayMode = CUPTI_KernelReplay;
    beginSession.maxRangesPerPass = maxRanges_;
    beginSession.maxLaunchesPerPass = maxRanges_;
    cupti::profilerBeginSession<true>(&beginSession);
    sessionBegun_ = true;

    CUpti_Profiler_SetConfig_Params setConfig = {
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    setConfig.ctx = context_;
    setConfig.pConfig = configImage_.data();
    setConfig.configSize = configImage_.size();
    setConfig.minNestingLevel = 1;
    setConfig.numNestingLevels = 1;
    setConfig.passIndex = 0;
    setConfig.targetNestingLevel = 1;
    cupti::profilerSetConfig<true>(&setConfig);
    configSet_ = true;

    CUpti_Profiler_EnableProfiling_Params enableProfiling = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    enableProfiling.ctx = context_;
    cupti::profilerEnableProfiling<true>(&enableProfiling);
    profilingEnabled_ = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_ = true;
    }
  } catch (const std::exception &error) {
    cleanup(&degradeReasons);
    appendDegrade(degradeReasons,
                  "NVIDIA NVPW hardware-counter profiling unavailable: " +
                      exceptionMessage(error));
  } catch (...) {
    cleanup(&degradeReasons);
    appendDegrade(degradeReasons,
                  "NVIDIA NVPW hardware-counter profiling unavailable: unknown "
                  "Perfworks/CUPTI error");
  }
}

void NvidiaHardwareCounters::recordKernel(const RuntimeTraceEventKey &event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_) {
    // FlagPrism: CUPTI KernelReplay can report the same logical launch once
    // per PM-counter pass.  Correlation IDs identify one launch, while a
    // graph launch may legitimately contain several nodes under that ID, so
    // include the graph node ID before suppressing a replay duplicate.
    const auto graphNodeId = [](const RuntimeTraceEventKey &candidate) {
      const auto it = candidate.vendorMetrics.find("graph_node_id");
      if (it == candidate.vendorMetrics.end()) {
        return uint64_t{0};
      }
      if (const auto *value = std::get_if<uint64_t>(&it->second)) {
        return *value;
      }
      return uint64_t{0};
    };
    if (event.correlationId != 0) {
      const auto duplicate = std::find_if(
          kernelEvents_.begin(), kernelEvents_.end(), [&](const auto &seen) {
            return seen.correlationId == event.correlationId &&
                   seen.opName == event.opName &&
                   seen.deviceId == event.deviceId &&
                   seen.streamId == event.streamId &&
                   graphNodeId(seen) == graphNodeId(event);
          });
      if (duplicate != kernelEvents_.end()) {
        return;
      }
    }
    kernelEvents_.push_back(event);
  }
}

void NvidiaHardwareCounters::evaluate(
    std::vector<RuntimeTraceEventKey> &vendorEvents,
    std::vector<std::string> &degradeReasons) {
  if (counterDataImage_.empty()) {
    appendDegrade(degradeReasons,
                  "NVIDIA NVPW hardware-counter profiling produced no "
                  "counter-data "
                  "image");
    return;
  }

  NVPW_CounterData_GetNumRanges_Params getRanges = {
      NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
  getRanges.pCounterDataImage = counterDataImage_.data();
  nvperfTarget::counterDataGetNumRanges<true>(&getRanges);
  if (getRanges.numRanges == 0) {
    appendDegrade(degradeReasons,
                  "NVIDIA NVPW hardware-counter profiling collected no kernel "
                  "ranges");
    return;
  }

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params evaluatorSize = {
      NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  evaluatorSize.pChipName = chipName_.c_str();
  evaluatorSize.pCounterAvailabilityImage = counterAvailability_.data();
  nvperf::cudaMetricsEvaluatorCalculateScratchBufferSize<true>(&evaluatorSize);
  std::vector<uint8_t> evaluatorScratch(evaluatorSize.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params evaluatorInitialize = {
      NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  evaluatorInitialize.pScratchBuffer = evaluatorScratch.data();
  evaluatorInitialize.scratchBufferSize = evaluatorScratch.size();
  evaluatorInitialize.pChipName = chipName_.c_str();
  evaluatorInitialize.pCounterAvailabilityImage = counterAvailability_.data();
  evaluatorInitialize.pCounterDataImage = counterDataImage_.data();
  evaluatorInitialize.counterDataImageSize = counterDataImage_.size();
  nvperf::cudaMetricsEvaluatorInitialize<true>(&evaluatorInitialize);
  auto *metricsEvaluator = evaluatorInitialize.pMetricsEvaluator;
  if (!metricsEvaluator) {
    throw std::runtime_error(
        "NVPW returned a null evaluation metrics evaluator");
  }

  try {
    std::vector<NVPW_MetricEvalRequest> requests;
    requests.reserve(metrics_.size());
    for (const auto &metric : metrics_) {
      NVPW_MetricEvalRequest request{};
      NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convert = {
          NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
      convert.pMetricsEvaluator = metricsEvaluator;
      convert.pMetricName = metric.evaluatorName.c_str();
      convert.pMetricEvalRequest = &request;
      convert.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
      nvperf::metricsEvaluatorConvertMetricNameToMetricEvalRequest<true>(
          &convert);
      requests.push_back(request);
    }

    NVPW_MetricsEvaluator_SetDeviceAttributes_Params setAttributes = {
        NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
    setAttributes.pMetricsEvaluator = metricsEvaluator;
    setAttributes.pCounterDataImage = counterDataImage_.data();
    setAttributes.counterDataImageSize = counterDataImage_.size();
    nvperf::metricsEvaluatorSetDeviceAttributes<true>(&setAttributes);

    std::vector<RuntimeTraceEventKey> kernelEvents;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      kernelEvents = kernelEvents_;
    }
    const auto rangesToEvaluate = std::min(getRanges.numRanges, maxRanges_);
    for (size_t rangeIndex = 0; rangeIndex < rangesToEvaluate; ++rangeIndex) {
      std::string rangeName = "range_" + std::to_string(rangeIndex);
      // FlagPrism: range names are diagnostic metadata only.  If a target
      // library lacks the optional description query, the metric association
      // remains usable through its AutoRange index and kernel event ordering.
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params descriptions = {
          NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE};
      descriptions.pCounterDataImage = counterDataImage_.data();
      descriptions.rangeIndex = rangeIndex;
      if (nvperfTarget::profilerCounterDataGetRangeDescriptions<false>(
              &descriptions) == NVPA_STATUS_SUCCESS &&
          descriptions.numDescriptions > 0) {
        std::vector<const char *> descriptionPtrs(descriptions.numDescriptions);
        descriptions.ppDescriptions = descriptionPtrs.data();
        if (nvperfTarget::profilerCounterDataGetRangeDescriptions<false>(
                &descriptions) == NVPA_STATUS_SUCCESS) {
          rangeName.clear();
          for (size_t i = 0; i < descriptions.numDescriptions; ++i) {
            if (i != 0) {
              rangeName += "/";
            }
            if (descriptionPtrs[i]) {
              rangeName += descriptionPtrs[i];
            }
          }
        }
      }

      std::vector<double> values(requests.size(), 0.0);
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateParams = {
          NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
      evaluateParams.pMetricsEvaluator = metricsEvaluator;
      evaluateParams.pMetricEvalRequests = requests.data();
      evaluateParams.numMetricEvalRequests = requests.size();
      evaluateParams.metricEvalRequestStructSize =
          NVPW_MetricEvalRequest_STRUCT_SIZE;
      evaluateParams.metricEvalRequestStrideSize =
          sizeof(NVPW_MetricEvalRequest);
      evaluateParams.pCounterDataImage = counterDataImage_.data();
      evaluateParams.counterDataImageSize = counterDataImage_.size();
      evaluateParams.rangeIndex = rangeIndex;
      evaluateParams.isolated = 1;
      evaluateParams.pMetricValues = values.data();
      nvperf::metricsEvaluatorEvaluateToGpuValues<true>(&evaluateParams);

      RuntimeTraceEventKey event;
      if (rangeIndex < kernelEvents.size()) {
        event = kernelEvents[rangeIndex];
      }
      if (event.opName.empty()) {
        event.opName = rangeName;
      }
      event.vendorMetrics["activity_kind"] = std::string("hardware_counter");
      event.vendorMetrics["hardware_counter_source"] = std::string("nvpw");
      event.vendorMetrics["hardware_range_index"] =
          static_cast<uint64_t>(rangeIndex);
      event.vendorMetrics["hardware_range_name"] = rangeName;
      for (size_t metricIndex = 0; metricIndex < metrics_.size();
           ++metricIndex) {
        event.vendorMetrics[metrics_[metricIndex].canonicalName] =
            values[metricIndex];
        if (metrics_[metricIndex].canonicalName == "occupancy") {
          // The selected NVPW metric is a percent-of-peak-sustained-active
          // metric; retain an explicit unit-bearing alias for JSON clients.
          event.vendorMetrics["occupancy_percent"] = values[metricIndex];
        } else if (metrics_[metricIndex].canonicalName == "throughput") {
          // FlagPrism: the throughput expression has the same percent unit,
          // but it is kept separate from achieved occupancy in the artifact.
          event.vendorMetrics["throughput_percent"] = values[metricIndex];
        } else if (metrics_[metricIndex].canonicalName == "bandwidth") {
          // FlagPrism: this is DRAM throughput as a percentage of the
          // sustained peak, not the activity-timestamp GB/s estimate exposed
          // by the separate `memory` metric.
          event.vendorMetrics["bandwidth_percent"] = values[metricIndex];
        }
      }
      vendorEvents.push_back(std::move(event));
    }
  } catch (...) {
    NVPW_MetricsEvaluator_Destroy_Params destroyEvaluator = {
        NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    destroyEvaluator.pMetricsEvaluator = metricsEvaluator;
    nvperf::metricsEvaluatorDestroy<false>(&destroyEvaluator);
    throw;
  }
  NVPW_MetricsEvaluator_Destroy_Params destroyEvaluator = {
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
  destroyEvaluator.pMetricsEvaluator = metricsEvaluator;
  nvperf::metricsEvaluatorDestroy<true>(&destroyEvaluator);
}

void NvidiaHardwareCounters::cleanup(std::vector<std::string> *degradeReasons) {
  const auto cleanupOne = [&](const char *phase, const auto &function) {
    try {
      function();
    } catch (const std::exception &error) {
      if (degradeReasons) {
        appendDegrade(*degradeReasons, "NVIDIA NVPW cleanup failed during " +
                                           std::string(phase) + ": " +
                                           error.what());
      }
    } catch (...) {
      if (degradeReasons) {
        appendDegrade(*degradeReasons, "NVIDIA NVPW cleanup failed during " +
                                           std::string(phase));
      }
    }
  };

  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = false;
  }
  if (profilingEnabled_) {
    CUpti_Profiler_DisableProfiling_Params disableProfiling = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    disableProfiling.ctx = context_;
    cleanupOne("disable profiling", [&]() {
      cupti::profilerDisableProfiling<true>(&disableProfiling);
    });
    profilingEnabled_ = false;
  }
  if (configSet_) {
    CUpti_Profiler_UnsetConfig_Params unsetConfig = {
        CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    unsetConfig.ctx = context_;
    cleanupOne("unset config",
               [&]() { cupti::profilerUnsetConfig<true>(&unsetConfig); });
    configSet_ = false;
  }
  if (sessionBegun_) {
    CUpti_Profiler_EndSession_Params endSession = {
        CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    endSession.ctx = context_;
    cleanupOne("end session",
               [&]() { cupti::profilerEndSession<true>(&endSession); });
    sessionBegun_ = false;
  }
  if (profilerInitialized_) {
    CUpti_Profiler_DeInitialize_Params deinitialize = {
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    cleanupOne("deinitialize profiler",
               [&]() { cupti::profilerDeInitialize<true>(&deinitialize); });
    profilerInitialized_ = false;
  }
}

void NvidiaHardwareCounters::stop(
    std::vector<RuntimeTraceEventKey> &vendorEvents,
    std::vector<std::string> &degradeReasons) {
  if (!profilerInitialized_ && !sessionBegun_ && !configSet_ &&
      !profilingEnabled_) {
    return;
  }

  // FlagPrism: disable and flush before NVPW evaluation.  KernelReplay writes
  // the decoded values into counterDataImage only after this flush completes.
  if (profilingEnabled_) {
    CUpti_Profiler_DisableProfiling_Params disableProfiling = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    disableProfiling.ctx = context_;
    try {
      cupti::profilerDisableProfiling<true>(&disableProfiling);
    } catch (const std::exception &error) {
      appendDegrade(degradeReasons, "NVIDIA NVPW disable profiling failed: " +
                                        std::string(error.what()));
    }
    profilingEnabled_ = false;
  }
  if (sessionBegun_) {
    CUpti_Profiler_FlushCounterData_Params flushCounterData = {
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    flushCounterData.ctx = context_;
    try {
      cupti::profilerFlushCounterData<true>(&flushCounterData);
      evaluate(vendorEvents, degradeReasons);
    } catch (const std::exception &error) {
      appendDegrade(degradeReasons,
                    "NVIDIA NVPW hardware-counter evaluation unavailable: " +
                        std::string(error.what()));
    } catch (...) {
      appendDegrade(
          degradeReasons,
          "NVIDIA NVPW hardware-counter evaluation unavailable: unknown "
          "Perfworks/CUPTI error");
    }
  }
  cleanup(&degradeReasons);
}

} // namespace proton
