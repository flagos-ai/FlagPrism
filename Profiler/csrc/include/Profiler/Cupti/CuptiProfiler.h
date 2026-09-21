#ifndef PROTON_PROFILER_CUPTI_PROFILER_H_
#define PROTON_PROFILER_CUPTI_PROFILER_H_

#include "Profiler/GPUProfiler.h"

#include "Data/Artifacts.h"

#include <string>
#include <vector>

namespace proton {

class CuptiProfiler : public GPUProfiler<CuptiProfiler> {
public:
  CuptiProfiler();
  virtual ~CuptiProfiler();

  // FlagPrism: the NVIDIA vendor adapter consumes the same CUPTI activity
  // records as the base profiler, avoiding a duplicate CUDA collector.
  void enableVendorEventCapture(bool enabled,
                                bool captureMemoryActivities = false);
  // FlagPrism: expose the existing CUPTI PC-sampling stream to the NVIDIA
  // vendor importer as typed associations, while keeping the legacy PC
  // SamplingMetric tree output intact.
  // NVIDIA vendor plans treat this optional capability as best effort; the
  // legacy `cupti:pcsampling` path keeps its existing fail-fast behavior.
  void setPCSamplingOptional(bool optional);
  // FlagPrism: configure the optional NVIDIA Perfworks range-profiler path.
  // The common CUPTI activity stream remains the source of base timings.
  void setHardwareCounterMetrics(const std::vector<std::string> &metricNames);
  void recordVendorPCSampling(size_t scopeId, const std::string &opName,
                              const std::string &stallMetricName,
                              uint64_t samples, uint64_t stalledSamples);
  std::vector<std::string> takeVendorRuntimeDegradeReasons();
  std::vector<RuntimeTraceEventKey> takeVendorRuntimeEvents();

private:
  struct CuptiProfilerPimpl;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
