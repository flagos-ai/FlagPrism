#include "Driver/GPU/CuptiApi.h"
#include "Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace cupti {

struct ExternLibCupti : public ExternLibBase {
  using RetType = CUptiResult;
  static constexpr const char *name = "libcupti.so";
  static inline std::string defaultDir = "";
  static constexpr RetType success = CUPTI_SUCCESS;
  static void *lib;
};

void *ExternLibCupti::lib = nullptr;

DEFINE_DISPATCH(ExternLibCupti, getVersion, cuptiGetVersion, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getContextId, cuptiGetContextId, CUcontext,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, activityRegisterCallbacks,
                cuptiActivityRegisterCallbacks,
                CUpti_BuffersCallbackRequestFunc,
                CUpti_BuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibCupti, subscribe, cuptiSubscribe,
                CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *)

DEFINE_DISPATCH(ExternLibCupti, enableDomain, cuptiEnableDomain, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain)

DEFINE_DISPATCH(ExternLibCupti, enableCallback, cuptiEnableCallback, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);

DEFINE_DISPATCH(ExternLibCupti, activityEnable, cuptiActivityEnable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisable, cuptiActivityDisable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityEnableContext,
                cuptiActivityEnableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisableContext,
                cuptiActivityDisableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityFlushAll, cuptiActivityFlushAll,
                uint32_t)

DEFINE_DISPATCH(ExternLibCupti, activityGetNextRecord,
                cuptiActivityGetNextRecord, uint8_t *, size_t,
                CUpti_Activity **)

DEFINE_DISPATCH(ExternLibCupti, activityPushExternalCorrelationId,
                cuptiActivityPushExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibCupti, activityPopExternalCorrelationId,
                cuptiActivityPopExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t *)

DEFINE_DISPATCH(ExternLibCupti, activitySetAttribute, cuptiActivitySetAttribute,
                CUpti_ActivityAttribute, size_t *, void *)

DEFINE_DISPATCH(ExternLibCupti, unsubscribe, cuptiUnsubscribe,
                CUpti_SubscriberHandle)

DEFINE_DISPATCH(ExternLibCupti, finalize, cuptiFinalize)

DEFINE_DISPATCH(ExternLibCupti, getGraphExecId, cuptiGetGraphExecId,
                CUgraphExec, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getGraphId, cuptiGetGraphId, CUgraph,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getCubinCrc, cuptiGetCubinCrc,
                CUpti_GetCubinCrcParams *);

DEFINE_DISPATCH(ExternLibCupti, getSassToSourceCorrelation,
                cuptiGetSassToSourceCorrelation,
                CUpti_GetSassToSourceCorrelationParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetNumStallReasons,
                cuptiPCSamplingGetNumStallReasons,
                CUpti_PCSamplingGetNumStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetStallReasons,
                cuptiPCSamplingGetStallReasons,
                CUpti_PCSamplingGetStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingSetConfigurationAttribute,
                cuptiPCSamplingSetConfigurationAttribute,
                CUpti_PCSamplingConfigurationInfoParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingEnable, cuptiPCSamplingEnable,
                CUpti_PCSamplingEnableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingDisable, cuptiPCSamplingDisable,
                CUpti_PCSamplingDisableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetData, cuptiPCSamplingGetData,
                CUpti_PCSamplingGetDataParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStart, cuptiPCSamplingStart,
                CUpti_PCSamplingStartParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStop, cuptiPCSamplingStop,
                CUpti_PCSamplingStopParams *);

DEFINE_DISPATCH(ExternLibCupti, profilerInitialize, cuptiProfilerInitialize,
                CUpti_Profiler_Initialize_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerDeInitialize, cuptiProfilerDeInitialize,
                CUpti_Profiler_DeInitialize_Params *);

DEFINE_DISPATCH(ExternLibCupti, deviceGetChipName, cuptiDeviceGetChipName,
                CUpti_Device_GetChipName_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerGetCounterAvailability,
                cuptiProfilerGetCounterAvailability,
                CUpti_Profiler_GetCounterAvailability_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerCounterDataImageCalculateSize,
                cuptiProfilerCounterDataImageCalculateSize,
                CUpti_Profiler_CounterDataImage_CalculateSize_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerCounterDataImageInitialize,
                cuptiProfilerCounterDataImageInitialize,
                CUpti_Profiler_CounterDataImage_Initialize_Params *);

DEFINE_DISPATCH(
    ExternLibCupti, profilerCounterDataImageCalculateScratchBufferSize,
    cuptiProfilerCounterDataImageCalculateScratchBufferSize,
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params *);

DEFINE_DISPATCH(
    ExternLibCupti, profilerCounterDataImageInitializeScratchBuffer,
    cuptiProfilerCounterDataImageInitializeScratchBuffer,
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerBeginSession, cuptiProfilerBeginSession,
                CUpti_Profiler_BeginSession_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerEndSession, cuptiProfilerEndSession,
                CUpti_Profiler_EndSession_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerSetConfig, cuptiProfilerSetConfig,
                CUpti_Profiler_SetConfig_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerUnsetConfig, cuptiProfilerUnsetConfig,
                CUpti_Profiler_UnsetConfig_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerEnableProfiling,
                cuptiProfilerEnableProfiling,
                CUpti_Profiler_EnableProfiling_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerDisableProfiling,
                cuptiProfilerDisableProfiling,
                CUpti_Profiler_DisableProfiling_Params *);

DEFINE_DISPATCH(ExternLibCupti, profilerFlushCounterData,
                cuptiProfilerFlushCounterData,
                CUpti_Profiler_FlushCounterData_Params *);

namespace nvperf {

struct ExternLibNvperfHost : public ExternLibBase {
  using RetType = NVPA_Status;
  static constexpr const char *name = "libnvperf_host.so";
  static inline std::string defaultDir = "";
  static constexpr RetType success = NVPA_STATUS_SUCCESS;
  static void *lib;
};

void *ExternLibNvperfHost::lib = nullptr;

DEFINE_DISPATCH(ExternLibNvperfHost, initializeHost, NVPW_InitializeHost,
                NVPW_InitializeHost_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost,
                cudaMetricsEvaluatorCalculateScratchBufferSize,
                NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize,
                NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, cudaMetricsEvaluatorInitialize,
                NVPW_CUDA_MetricsEvaluator_Initialize,
                NVPW_CUDA_MetricsEvaluator_Initialize_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, metricsEvaluatorDestroy,
                NVPW_MetricsEvaluator_Destroy,
                NVPW_MetricsEvaluator_Destroy_Params *);

DEFINE_DISPATCH(
    ExternLibNvperfHost, metricsEvaluatorConvertMetricNameToMetricEvalRequest,
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest,
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, metricsEvaluatorGetMetricRawDependencies,
                NVPW_MetricsEvaluator_GetMetricRawDependencies,
                NVPW_MetricsEvaluator_GetMetricRawDependencies_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, metricsEvaluatorSetDeviceAttributes,
                NVPW_MetricsEvaluator_SetDeviceAttributes,
                NVPW_MetricsEvaluator_SetDeviceAttributes_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, metricsEvaluatorEvaluateToGpuValues,
                NVPW_MetricsEvaluator_EvaluateToGpuValues,
                NVPW_MetricsEvaluator_EvaluateToGpuValues_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, cudaRawMetricsConfigCreateV2,
                NVPW_CUDA_RawMetricsConfig_Create_V2,
                NVPW_CUDA_RawMetricsConfig_Create_V2_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigSetCounterAvailability,
                NVPW_RawMetricsConfig_SetCounterAvailability,
                NVPW_RawMetricsConfig_SetCounterAvailability_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigBeginPassGroup,
                NVPW_RawMetricsConfig_BeginPassGroup,
                NVPW_RawMetricsConfig_BeginPassGroup_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigAddMetrics,
                NVPW_RawMetricsConfig_AddMetrics,
                NVPW_RawMetricsConfig_AddMetrics_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigEndPassGroup,
                NVPW_RawMetricsConfig_EndPassGroup,
                NVPW_RawMetricsConfig_EndPassGroup_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigGenerateConfigImage,
                NVPW_RawMetricsConfig_GenerateConfigImage,
                NVPW_RawMetricsConfig_GenerateConfigImage_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigGetConfigImage,
                NVPW_RawMetricsConfig_GetConfigImage,
                NVPW_RawMetricsConfig_GetConfigImage_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, rawMetricsConfigDestroy,
                NVPW_RawMetricsConfig_Destroy,
                NVPW_RawMetricsConfig_Destroy_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, cudaCounterDataBuilderCreate,
                NVPW_CUDA_CounterDataBuilder_Create,
                NVPW_CUDA_CounterDataBuilder_Create_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, counterDataBuilderAddMetrics,
                NVPW_CounterDataBuilder_AddMetrics,
                NVPW_CounterDataBuilder_AddMetrics_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, counterDataBuilderGetCounterDataPrefix,
                NVPW_CounterDataBuilder_GetCounterDataPrefix,
                NVPW_CounterDataBuilder_GetCounterDataPrefix_Params *);

DEFINE_DISPATCH(ExternLibNvperfHost, counterDataBuilderDestroy,
                NVPW_CounterDataBuilder_Destroy,
                NVPW_CounterDataBuilder_Destroy_Params *);

void setLibPath(const std::string &path) {
  ExternLibNvperfHost::defaultDir = path;
}

} // namespace nvperf

namespace nvperfTarget {

struct ExternLibNvperfTarget : public ExternLibBase {
  using RetType = NVPA_Status;
  static constexpr const char *name = "libnvperf_target.so";
  static inline std::string defaultDir = "";
  static constexpr RetType success = NVPA_STATUS_SUCCESS;
  static void *lib;
};

void *ExternLibNvperfTarget::lib = nullptr;

DEFINE_DISPATCH(ExternLibNvperfTarget, counterDataGetNumRanges,
                NVPW_CounterData_GetNumRanges,
                NVPW_CounterData_GetNumRanges_Params *);

DEFINE_DISPATCH(ExternLibNvperfTarget, profilerCounterDataGetRangeDescriptions,
                NVPW_Profiler_CounterData_GetRangeDescriptions,
                NVPW_Profiler_CounterData_GetRangeDescriptions_Params *);

void setLibPath(const std::string &path) {
  ExternLibNvperfTarget::defaultDir = path;
}

} // namespace nvperfTarget

void setLibPath(const std::string &path) { ExternLibCupti::defaultDir = path; }

} // namespace cupti

} // namespace proton
