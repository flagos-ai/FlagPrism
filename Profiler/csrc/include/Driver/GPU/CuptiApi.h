#ifndef PROTON_DRIVER_GPU_CUPTI_H_
#define PROTON_DRIVER_GPU_CUPTI_H_

#include "cupti.h"
#include "cupti_pcsampling.h"
#include "cupti_profiler_target.h"
#include "cupti_target.h"
#include "nvperf_cuda_host.h"
#include "nvperf_host.h"
#include "nvperf_target.h"
#include <string>

namespace proton {

namespace cupti {

template <bool CheckSuccess> CUptiResult getVersion(uint32_t *version);

template <bool CheckSuccess>
CUptiResult getContextId(CUcontext context, uint32_t *pCtxId);

template <bool CheckSuccess>
CUptiResult activityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

template <bool CheckSuccess>
CUptiResult subscribe(CUpti_SubscriberHandle *subscriber,
                      CUpti_CallbackFunc callback, void *userdata);

template <bool CheckSuccess>
CUptiResult enableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                         CUpti_CallbackDomain domain);

template <bool CheckSuccess>
CUptiResult enableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain, CUpti_CallbackId cbid);

template <bool CheckSuccess>
CUptiResult activityEnableContext(CUcontext context, CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityDisableContext(CUcontext context, CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityEnable(CUpti_ActivityKind kind);

template <bool CheckSuccess>
CUptiResult activityDisable(CUpti_ActivityKind kind);

template <bool CheckSuccess> CUptiResult activityFlushAll(uint32_t flag);

template <bool CheckSuccess>
CUptiResult activityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes,
                                  CUpti_Activity **record);

template <bool CheckSuccess>
CUptiResult
activityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                  uint64_t id);

template <bool CheckSuccess>
CUptiResult activityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                             uint64_t *lastId);

template <bool CheckSuccess>
CUptiResult activitySetAttribute(CUpti_ActivityAttribute attr,
                                 size_t *valueSize, void *value);

template <bool CheckSuccess>
CUptiResult unsubscribe(CUpti_SubscriberHandle subscriber);

template <bool CheckSuccess> CUptiResult finalize();

template <bool CheckSuccess>
CUptiResult getGraphExecId(CUgraphExec graph, uint32_t *pId);

template <bool CheckSuccess>
CUptiResult getGraphId(CUgraph graph, uint32_t *pId);

template <bool CheckSuccess>
CUptiResult getCubinCrc(CUpti_GetCubinCrcParams *pParams);

template <bool CheckSuccess>
CUptiResult
getSassToSourceCorrelation(CUpti_GetSassToSourceCorrelationParams *pParams);

template <bool CheckSuccess>
CUptiResult
pcSamplingGetNumStallReasons(CUpti_PCSamplingGetNumStallReasonsParams *pParams);

template <bool CheckSuccess>
CUptiResult
pcSamplingGetStallReasons(CUpti_PCSamplingGetStallReasonsParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingSetConfigurationAttribute(
    CUpti_PCSamplingConfigurationInfoParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingEnable(CUpti_PCSamplingEnableParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingDisable(CUpti_PCSamplingDisableParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingGetData(CUpti_PCSamplingGetDataParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingStart(CUpti_PCSamplingStartParams *pParams);

template <bool CheckSuccess>
CUptiResult pcSamplingStop(CUpti_PCSamplingStopParams *pParams);

// FlagPrism: CUPTI's range-profiler entry points are dispatched just like the
// existing activity and PC-sampling APIs.  Keeping them dynamically loaded
// lets the NVIDIA adapter retain the packaged-CUPTI path and degrade cleanly
// when a driver does not permit performance-counter access.
template <bool CheckSuccess>
CUptiResult profilerInitialize(CUpti_Profiler_Initialize_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerDeInitialize(CUpti_Profiler_DeInitialize_Params *pParams);

template <bool CheckSuccess>
CUptiResult deviceGetChipName(CUpti_Device_GetChipName_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerCounterDataImageCalculateSize(
    CUpti_Profiler_CounterDataImage_CalculateSize_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerCounterDataImageInitialize(
    CUpti_Profiler_CounterDataImage_Initialize_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerCounterDataImageCalculateScratchBufferSize(
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerCounterDataImageInitializeScratchBuffer(
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerBeginSession(CUpti_Profiler_BeginSession_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerEndSession(CUpti_Profiler_EndSession_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerSetConfig(CUpti_Profiler_SetConfig_Params *pParams);

template <bool CheckSuccess>
CUptiResult profilerUnsetConfig(CUpti_Profiler_UnsetConfig_Params *pParams);

template <bool CheckSuccess>
CUptiResult
profilerEnableProfiling(CUpti_Profiler_EnableProfiling_Params *pParams);

template <bool CheckSuccess>
CUptiResult
profilerDisableProfiling(CUpti_Profiler_DisableProfiling_Params *pParams);

template <bool CheckSuccess>
CUptiResult
profilerFlushCounterData(CUpti_Profiler_FlushCounterData_Params *pParams);

// FlagPrism: NVPW host/target calls are also dispatched rather than linked
// directly.  This avoids coupling the profiler extension to one CUDA toolkit
// while still using the NVIDIA Perfworks implementation when it is present.
namespace nvperf {

template <bool CheckSuccess>
NVPA_Status initializeHost(NVPW_InitializeHost_Params *pParams);

template <bool CheckSuccess>
NVPA_Status cudaMetricsEvaluatorCalculateScratchBufferSize(
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params *pParams);

template <bool CheckSuccess>
NVPA_Status cudaMetricsEvaluatorInitialize(
    NVPW_CUDA_MetricsEvaluator_Initialize_Params *pParams);

template <bool CheckSuccess>
NVPA_Status
metricsEvaluatorDestroy(NVPW_MetricsEvaluator_Destroy_Params *pParams);

template <bool CheckSuccess>
NVPA_Status metricsEvaluatorConvertMetricNameToMetricEvalRequest(
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params *pParams);

template <bool CheckSuccess>
NVPA_Status metricsEvaluatorGetMetricRawDependencies(
    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params *pParams);

template <bool CheckSuccess>
NVPA_Status metricsEvaluatorSetDeviceAttributes(
    NVPW_MetricsEvaluator_SetDeviceAttributes_Params *pParams);

template <bool CheckSuccess>
NVPA_Status metricsEvaluatorEvaluateToGpuValues(
    NVPW_MetricsEvaluator_EvaluateToGpuValues_Params *pParams);

template <bool CheckSuccess>
NVPA_Status cudaRawMetricsConfigCreateV2(
    NVPW_CUDA_RawMetricsConfig_Create_V2_Params *pParams);

template <bool CheckSuccess>
NVPA_Status rawMetricsConfigSetCounterAvailability(
    NVPW_RawMetricsConfig_SetCounterAvailability_Params *pParams);

template <bool CheckSuccess>
NVPA_Status rawMetricsConfigBeginPassGroup(
    NVPW_RawMetricsConfig_BeginPassGroup_Params *pParams);

template <bool CheckSuccess>
NVPA_Status
rawMetricsConfigAddMetrics(NVPW_RawMetricsConfig_AddMetrics_Params *pParams);

template <bool CheckSuccess>
NVPA_Status rawMetricsConfigEndPassGroup(
    NVPW_RawMetricsConfig_EndPassGroup_Params *pParams);

template <bool CheckSuccess>
NVPA_Status rawMetricsConfigGenerateConfigImage(
    NVPW_RawMetricsConfig_GenerateConfigImage_Params *pParams);

template <bool CheckSuccess>
NVPA_Status rawMetricsConfigGetConfigImage(
    NVPW_RawMetricsConfig_GetConfigImage_Params *pParams);

template <bool CheckSuccess>
NVPA_Status
rawMetricsConfigDestroy(NVPW_RawMetricsConfig_Destroy_Params *pParams);

template <bool CheckSuccess>
NVPA_Status cudaCounterDataBuilderCreate(
    NVPW_CUDA_CounterDataBuilder_Create_Params *pParams);

template <bool CheckSuccess>
NVPA_Status counterDataBuilderAddMetrics(
    NVPW_CounterDataBuilder_AddMetrics_Params *pParams);

template <bool CheckSuccess>
NVPA_Status counterDataBuilderGetCounterDataPrefix(
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params *pParams);

template <bool CheckSuccess>
NVPA_Status
counterDataBuilderDestroy(NVPW_CounterDataBuilder_Destroy_Params *pParams);

void setLibPath(const std::string &path);

} // namespace nvperf

namespace nvperfTarget {

void setLibPath(const std::string &path);

template <bool CheckSuccess>
NVPA_Status
counterDataGetNumRanges(NVPW_CounterData_GetNumRanges_Params *pParams);

template <bool CheckSuccess>
NVPA_Status profilerCounterDataGetRangeDescriptions(
    NVPW_Profiler_CounterData_GetRangeDescriptions_Params *pParams);

} // namespace nvperfTarget

void setLibPath(const std::string &path);

} // namespace cupti

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
