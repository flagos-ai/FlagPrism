#ifndef PROTON_PROFILER_VENDOR_NVIDIA_ADAPTER_H_
#define PROTON_PROFILER_VENDOR_NVIDIA_ADAPTER_H_

#include "Profiler/Vendor/Adapter.h"

namespace proton {

class NvidiaMetricsImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override;

  VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                               const VendorProfilePlan &plan) const override;
};

class NvidiaAdapter final : public VendorAdapter {
public:
  static const NvidiaAdapter &instance();

  std::string getName() const override;
  DeviceType getDeviceType() const override;
  std::vector<std::string> getSupportedVendorMetrics() const override;
  VendorProfilePlan
  makePlan(const VendorProfileOptions &options) const override;
  void configureRuntimeProfiler(const std::string &profilerPath,
                                bool captureVendorEvents) const override;
  void configureRuntimeProfiler(const std::string &profilerPath,
                                const VendorProfilePlan &plan) const override;
  Profiler *getRuntimeProfiler() const override;
  std::unique_ptr<VendorMetricsImporter> createImporter() const override;

private:
  NvidiaAdapter() = default;
};

} // namespace proton

#endif // PROTON_PROFILER_VENDOR_NVIDIA_ADAPTER_H_
