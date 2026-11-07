#ifndef FLAGPRISM_HCU_PROFILER_H
#define FLAGPRISM_HCU_PROFILER_H
#include "Profiler/Vendor/Adapter.h"
#include "Utility/Singleton.h"
namespace proton {
class HcuAdapter final : public VendorAdapter, public Singleton<HcuAdapter> {
public:
  std::string getName() const override { return "hcu"; }
  DeviceType getDeviceType() const override { return DeviceType::HIP; }
  std::vector<std::string> getSupportedVendorMetrics() const override {
    return {"launch_stats", "resource_usage", "kernel_duration",
            "theoretical_occupancy"};
  }
  VendorProfilePlan makePlan(const VendorProfileOptions &) const override;
  Profiler *getRuntimeProfiler() const override;
  void configureRuntimeProfiler(const std::string &,
                                const VendorProfilePlan &) const override;
  std::unique_ptr<VendorMetricsImporter> createImporter() const override;
};
} // namespace proton
#endif
