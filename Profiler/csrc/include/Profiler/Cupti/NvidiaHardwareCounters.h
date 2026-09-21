#ifndef PROTON_PROFILER_CUPTI_NVIDIA_HARDWARE_COUNTERS_H_
#define PROTON_PROFILER_CUPTI_NVIDIA_HARDWARE_COUNTERS_H_

#include "Data/Artifacts.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/CuptiApi.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace proton {

// FlagPrism: this helper owns the NVIDIA range-profiler lifetime.  It uses
// CUPTI for device/session control and NVPW/Perfworks for metric scheduling and
// evaluation, so the common profiler can expose real PM-counter values without
// introducing a second profiling process.
class NvidiaHardwareCounters {
public:
  NvidiaHardwareCounters() = default;
  ~NvidiaHardwareCounters();

  // The canonical names are intentionally small and backend-neutral.  The
  // implementation maps them to an architecture-specific NVPW metric request.
  void configure(const std::vector<std::string> &metricNames,
                 size_t maxRanges = 4096);
  bool isConfigured() const;
  bool isActive() const;

  // Best-effort by design: an unavailable performance-counter permission must
  // not prevent CUPTI activity profiling from producing its base artifacts.
  void start(CUcontext context, std::vector<std::string> &degradeReasons);

  // Called from the CUPTI activity callback before the common correlation map
  // consumes the activity.  Events are matched to AutoRange indices in order.
  void recordKernel(const RuntimeTraceEventKey &event);

  // Stops the range session, evaluates every collected range, and appends
  // typed hardware-counter events to the shared vendor event queue.
  void stop(std::vector<RuntimeTraceEventKey> &vendorEvents,
            std::vector<std::string> &degradeReasons);

private:
  struct MetricSpec {
    std::string canonicalName;
    std::string evaluatorName;
  };

  bool configured_{false};
  bool active_{false};
  bool profilerInitialized_{false};
  bool sessionBegun_{false};
  bool configSet_{false};
  bool profilingEnabled_{false};
  CUcontext context_{nullptr};
  size_t maxRanges_{4096};
  std::string chipName_{};
  std::vector<MetricSpec> metrics_{};
  std::vector<RuntimeTraceEventKey> kernelEvents_{};

  // These buffers must remain alive from BeginSession through evaluation.
  std::vector<uint8_t> counterAvailability_{};
  std::vector<uint8_t> configImage_{};
  std::vector<uint8_t> counterDataPrefix_{};
  std::vector<uint8_t> counterDataImage_{};
  std::vector<uint8_t> counterDataScratchBuffer_{};

  mutable std::mutex mutex_{};

  static void appendDegrade(std::vector<std::string> &degradeReasons,
                            const std::string &reason);
  void cleanup(std::vector<std::string> *degradeReasons);
  void evaluate(std::vector<RuntimeTraceEventKey> &vendorEvents,
                std::vector<std::string> &degradeReasons);
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_NVIDIA_HARDWARE_COUNTERS_H_
