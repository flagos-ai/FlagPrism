// FlagPrism：MCPTI runtime adapter using the common vendor artifact contract.
#ifndef FLAGPRISM_METAX_PROFILER_H
#define FLAGPRISM_METAX_PROFILER_H
#include "Profiler/Vendor/Adapter.h"
#include "Utility/Singleton.h"
namespace proton {
class MetaxAdapter final : public VendorAdapter,
                           public Singleton<MetaxAdapter> {
public:
  std::string getName() const override { return "metax"; }
  DeviceType getDeviceType() const override { return DeviceType::METAX; }
  std::vector<std::string> getSupportedVendorMetrics() const override {
    return {
        "launch_stats",          "resource_usage",        "kernel_duration",
        "theoretical_occupancy", "dram_read_bytes",       "dram_write_bytes",
        "dram_read_throughput",  "dram_write_throughput", "inst_executed"};
  }
  VendorProfilePlan makePlan(const VendorProfileOptions &) const override;
  Profiler *getRuntimeProfiler() const override;
  void configureRuntimeProfiler(const std::string &,
                                const VendorProfilePlan &) const override;
  std::unique_ptr<VendorMetricsImporter> createImporter() const override;
};
} // namespace proton
#endif

// FlagPrism end
