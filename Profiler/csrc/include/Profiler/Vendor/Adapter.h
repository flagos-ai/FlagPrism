#ifndef PROTON_PROFILER_VENDOR_ADAPTER_H_
#define PROTON_PROFILER_VENDOR_ADAPTER_H_

#include "Data/Artifacts.h"
#include "Profiler/Vendor/Mode.h"

#include <memory>
#include <string>
#include <vector>

namespace proton {

class Profiler;

// Imports adapter-specific profiler output, such as aclprof/msprof exports.
class VendorMetricsImporter {
public:
  virtual ~VendorMetricsImporter() = default;

  virtual std::string getName() const = 0;

  virtual VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                                       const VendorProfilePlan &plan) const = 0;
};

// Adapter contract for vendor runtime backends such as CANN and Tianshu.
class VendorAdapter {
public:
  virtual ~VendorAdapter() = default;

  virtual std::string getName() const = 0;

  virtual DeviceType getDeviceType() const = 0;

  virtual std::vector<std::string> getSupportedVendorMetrics() const = 0;

  virtual VendorProfilePlan
  makePlan(const VendorProfileOptions &options) const = 0;

  // The concrete runtime profiler is still expected to inherit
  // proton::Profiler. Returning nullptr is acceptable for an unfinished adapter
  // skeleton.
  virtual Profiler *getRuntimeProfiler() const = 0;

  // Vendor adapters that use a dynamically loaded runtime may consume the
  // path selected by the public Python API and the session's capture choice.
  // Existing adapters keep the no-op default so this remains source-compatible
  // with their implementations.
  virtual void configureRuntimeProfiler(const std::string &profilerPath,
                                        bool captureVendorEvents) const {
    (void)profilerPath;
    (void)captureVendorEvents;
  }

  // FlagPrism: adapters that share a runtime collector can use the complete
  // plan to enable only the activity classes required by the request. The
  // boolean overload remains the compatibility path for existing adapters.
  virtual void configureRuntimeProfiler(const std::string &profilerPath,
                                        const VendorProfilePlan &plan) const {
    configureRuntimeProfiler(profilerPath, !plan.enabledVendorMetrics.empty());
  }

  virtual std::unique_ptr<VendorMetricsImporter> createImporter() const = 0;
};

class VendorAdapterRegistry {
public:
  static const VendorAdapter *find(const std::string &name);

  static std::vector<std::string> names();
};

} // namespace proton

#endif // PROTON_PROFILER_VENDOR_ADAPTER_H_
