#include "Profiler/Vendor/TianshuAdapter.h"

#include "Profiler/Vendor/TianshuProfiler.h"
#include "Utility/String.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace proton {
namespace {

std::string join(const std::vector<std::string> &items) {
  std::ostringstream os;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << items[i];
  }
  return os.str();
}

bool contains(const std::vector<std::string> &items, const std::string &value) {
  return std::find(items.begin(), items.end(), value) != items.end();
}

} // namespace

std::string TianshuMetricsImporter::getName() const { return "ixkn_importer"; }

VendorProfileArtifact TianshuMetricsImporter::import(
    const SessionProfileMetadata &metadata,
    const VendorProfilePlan &plan) const {
  auto artifact = TianshuProfiler::importIxknOutput(metadata, plan);
  artifact.backend = metadata.backend;
  artifact.importer = getName();
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;
  artifact.degradeReasons.insert(artifact.degradeReasons.begin(),
                                 plan.degradeReasons.begin(),
                                 plan.degradeReasons.end());
  if (artifact.associations.empty() && artifact.degradeReasons.empty()) {
    artifact.degradeReasons.push_back(
        "No Tianshu ixKN profiling associations could be imported.");
  }
  return artifact;
}

const TianshuAdapter &TianshuAdapter::instance() {
  static const TianshuAdapter adapter;
  return adapter;
}

std::string TianshuAdapter::getName() const { return "tianshu"; }

DeviceType TianshuAdapter::getDeviceType() const {
  return DeviceType::TIANSHU;
}

std::vector<std::string> TianshuAdapter::getSupportedVendorMetrics() const {
  // These names correspond to the ixKN sections documented by Tianshu.
  return {"launch_stats", "occupancy", "instruction", "memory"};
}

VendorProfilePlan
TianshuAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = true;

  auto &requested = plan.requested;
  if (requested.adapterOptions.count("ixkn_external") == 0) {
    requested.adapterOptions["ixkn_external"] = "true";
  }
  if (requested.adapterOptions.count("ixkn_sections") == 0) {
    requested.adapterOptions["ixkn_sections"] =
        "LaunchStats,Occupancy,Instruction,Memory";
  }
  if (requested.adapterOptions.count("ixkn_import_path") == 0) {
    const char *env = std::getenv("FLAGTREE_PROFILER_TIANSHU_IMPORT_PATH");
    if (env && *env) {
      requested.adapterOptions["ixkn_import_path"] = env;
    }
  }

  const auto supported = getSupportedVendorMetrics();
  for (const auto &request : requested.vendorMetrics) {
    auto metric = toLower(trim(request.name));
    if (metric == "launchstats") {
      metric = "launch_stats";
    }
    if (contains(supported, metric)) {
      if (!contains(plan.enabledVendorMetrics, metric)) {
        plan.enabledVendorMetrics.push_back(metric);
      }
    } else {
      plan.disabledVendorMetrics.push_back(request.name);
    }
  }
  if (!plan.disabledVendorMetrics.empty()) {
    plan.degradeReasons.push_back(
        "Unsupported Tianshu ixKN metrics: " +
        join(plan.disabledVendorMetrics));
  }

  if (requested.adapterOptions.count("ixkn_import_path") == 0 &&
      requested.adapterOptions.count("ixkn_output") == 0 &&
      requested.adapterOptions.count("output_path") == 0) {
    plan.degradeReasons.push_back(
        "Tianshu ixKN is an external process profiler; provide "
        "ixkn_import_path or run the FlagTree ixKN wrapper for vendor data.");
  }
  return plan;
}

Profiler *TianshuAdapter::getRuntimeProfiler() const {
  return &TianshuProfiler::instance();
}

std::unique_ptr<VendorMetricsImporter>
TianshuAdapter::createImporter() const {
  return std::make_unique<TianshuMetricsImporter>();
}

} // namespace proton
