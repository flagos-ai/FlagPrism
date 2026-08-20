#ifndef PROTON_PROFILER_TIANSHU_PROFILER_H_
#define PROTON_PROFILER_TIANSHU_PROFILER_H_

#include "Context/Context.h"
#include "Data/Artifacts.h"
#include "Profiler/Profiler.h"
#include "Profiler/Vendor/Mode.h"
#include "Utility/Singleton.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace proton {

class TianshuProfiler final : public Profiler,
                              public OpInterface,
                              public Singleton<TianshuProfiler> {
public:
  TianshuProfiler() = default;
  ~TianshuProfiler() override = default;

  static VendorProfileArtifact
  importIxknOutput(const SessionProfileMetadata &metadata,
                   const VendorProfilePlan &plan);

private:
  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;

  void doStart() override;
  void doFlush() override;
  void doStop() override;
  void doSetMode(const std::vector<std::string> &modeAndOptions) override;

  std::vector<RuntimeTraceEventKey> drainRuntimeEvents();

  std::mutex mutex;
  std::unordered_map<size_t, uint64_t> opStartTimesNs;
  std::vector<RuntimeTraceEventKey> runtimeEvents;
  std::vector<std::string> runtimeDegradeReasons;
  uint32_t deviceId = 0;
  std::string importPath;
};

} // namespace proton

#endif // PROTON_PROFILER_TIANSHU_PROFILER_H_
