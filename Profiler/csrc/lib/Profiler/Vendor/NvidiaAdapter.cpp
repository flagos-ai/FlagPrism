#include "Profiler/Vendor/NvidiaAdapter.h"

#include "Profiler/Cupti/CuptiProfiler.h"
#include "Utility/String.h"

#include <algorithm>
#include <map>
#include <sstream>

namespace proton {
namespace {

std::string join(const std::vector<std::string> &items) {
  std::ostringstream stream;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      stream << ",";
    }
    stream << items[i];
  }
  return stream.str();
}

bool contains(const std::vector<std::string> &items, const std::string &value) {
  return std::find(items.begin(), items.end(), value) != items.end();
}

bool optionEnabled(const std::map<std::string, std::string> &options,
                   const std::string &name) {
  const auto it = options.find(name);
  if (it == options.end()) {
    return false;
  }
  const auto value = toLower(trim(it->second));
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

} // namespace

std::string NvidiaMetricsImporter::getName() const { return "cupti_importer"; }

VendorProfileArtifact
NvidiaMetricsImporter::import(const SessionProfileMetadata &metadata,
                              const VendorProfilePlan &plan) const {
  VendorProfileArtifact artifact;
  artifact.backend = metadata.backend;
  artifact.importer = getName();
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;
  artifact.degradeReasons = plan.degradeReasons;

  // FlagPrism: a driver may reject optional PC sampling after the session has
  // already started. Preserve the base CUPTI output and surface that loss in
  // the same vendor artifact instead of failing the whole profile.
  auto runtimeDegradeReasons =
      CuptiProfiler::instance().takeVendorRuntimeDegradeReasons();
  artifact.degradeReasons.insert(artifact.degradeReasons.end(),
                                 runtimeDegradeReasons.begin(),
                                 runtimeDegradeReasons.end());

  // FlagPrism: CUPTI already produces the base KernelMetric stream. The
  // importer converts the retained activity keys into vendor associations,
  // keeping launch_stats and kernel_duration usable by the common artifact
  // and correlation pipeline without requiring a second profiler process.
  for (const auto &event :
       CuptiProfiler::instance().takeVendorRuntimeEvents()) {
    const auto activityKindIt = event.vendorMetrics.find("activity_kind");
    const auto *activityKind =
        activityKindIt == event.vendorMetrics.end()
            ? nullptr
            : std::get_if<std::string>(&activityKindIt->second);
    const bool isMemoryActivity = activityKind && (*activityKind == "memcpy" ||
                                                   *activityKind == "memset");
    const bool isPCSamplingActivity =
        activityKind && *activityKind == "pcsampling";
    const bool isHardwareCounterActivity =
        activityKind && *activityKind == "hardware_counter";
    const bool isKernelActivity = !isMemoryActivity && !isPCSamplingActivity &&
                                  !isHardwareCounterActivity;
    const bool wantsKernelActivity =
        contains(plan.enabledVendorMetrics, "launch_stats") ||
        contains(plan.enabledVendorMetrics, "kernel_duration");
    const bool wantsInstruction =
        contains(plan.enabledVendorMetrics, "instruction");
    const bool wantsHardwareCounters =
        contains(plan.enabledVendorMetrics, "occupancy") ||
        contains(plan.enabledVendorMetrics, "throughput") ||
        contains(plan.enabledVendorMetrics, "bandwidth") ||
        contains(plan.enabledVendorMetrics, "instruction_count");
    if ((isMemoryActivity && !contains(plan.enabledVendorMetrics, "memory")) ||
        (isPCSamplingActivity && !wantsInstruction) ||
        (isHardwareCounterActivity && !wantsHardwareCounters) ||
        (isKernelActivity && !wantsKernelActivity)) {
      continue;
    }

    VendorMetricAssociation association;
    association.runtimeEvent = event;
    association.state = VendorMetricState::Collected;
    association.source = isMemoryActivity       ? "cupti_activity_memory"
                         : isPCSamplingActivity ? "cupti_pc_sampling"
                         : isHardwareCounterActivity
                             ? "nvidia_nvpw_hardware_counter"
                             : "cupti_activity";
    association.note =
        isMemoryActivity       ? "Collected from NVIDIA CUPTI memcpy/memset "
                                 "activity records; memory_bytes and the derived "
                                 "transfer rate are not DRAM hardware counters."
        : isPCSamplingActivity ? "Collected from NVIDIA CUPTI PC sampling; "
                                 "instruction_samples and stall buckets are "
                                 "sampling observations, not PM counters."
        : isHardwareCounterActivity ? "Collected from NVIDIA NVPW/Perfworks "
                                      "range profiling; values are evaluated "
                                      "PM-counter metrics for the AutoRange "
                                      "kernel."
                                    : "Collected from NVIDIA CUPTI kernel "
                                      "activity records; launch_stats is one "
                                      "activity record, not a hardware "
                                      "counter.";
    if (event.scopeId == 0) {
      association.note +=
          " A synthetic scope is used because CUPTI reported a graph or "
          "external CUDA kernel whose child scope is created while importing.";
    }
    if (isKernelActivity &&
        contains(plan.enabledVendorMetrics, "launch_stats")) {
      association.metrics["launch_stats"] = static_cast<uint64_t>(1);
      // FlagPrism: launch_stats includes the launch geometry and resource
      // properties copied from the same CUPTI activity record.  Keeping them
      // as separate typed fields makes them consumable by both JSON clients
      // and the common FlagTree metric overlay.
      for (const auto &[name, value] : event.vendorMetrics) {
        if (name != "activity_kind") {
          association.metrics[name] = value;
        }
      }
    }
    if (isKernelActivity &&
        contains(plan.enabledVendorMetrics, "kernel_duration")) {
      association.metrics["kernel_duration_us"] =
          static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
    }
    if (isMemoryActivity) {
      for (const auto &[name, value] : event.vendorMetrics) {
        if (name != "activity_kind") {
          association.metrics[name] = value;
        }
      }
      association.metrics["memory_duration_us"] =
          static_cast<double>(event.endTimeNs - event.startTimeNs) / 1000.0;
    }
    if (isPCSamplingActivity) {
      // FlagPrism: copy the sampled instruction buckets without adding a
      // synthetic duration, because PC samples do not carry an activity
      // interval of their own.
      for (const auto &[name, value] : event.vendorMetrics) {
        if (name != "activity_kind") {
          association.metrics[name] = value;
        }
      }
    }
    if (isHardwareCounterActivity) {
      // FlagPrism: NVPW events already carry the evaluated metric values. Do
      // not synthesize a second duration or alter the counter units.
      for (const auto &[name, value] : event.vendorMetrics) {
        if (name != "activity_kind") {
          association.metrics[name] = value;
        }
      }
    }
    artifact.associations.push_back(std::move(association));
  }

  if (artifact.associations.empty() && !plan.enabledVendorMetrics.empty()) {
    artifact.degradeReasons.push_back(
        "No NVIDIA CUPTI activity associations were collected.");
  }
  return artifact;
}

const NvidiaAdapter &NvidiaAdapter::instance() {
  static const NvidiaAdapter adapter;
  return adapter;
}

std::string NvidiaAdapter::getName() const { return "nvidia"; }

DeviceType NvidiaAdapter::getDeviceType() const { return DeviceType::CUDA; }

std::vector<std::string> NvidiaAdapter::getSupportedVendorMetrics() const {
  // FlagPrism: activity metrics remain available alongside the real NVPW
  // occupancy/throughput/bandwidth/instruction_count counters. `memory` is
  // transfer activity (bytes/timestamps), and `instruction` is CUPTI PC
  // sampling; the four hardware metrics are evaluated through the NVIDIA
  // range-profiler API.
  return {"launch_stats", "kernel_duration", "memory",    "instruction",
          "occupancy",    "throughput",      "bandwidth", "instruction_count"};
}

VendorProfilePlan
NvidiaAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = true;

  if (!options.runtimeBaseEnabled) {
    plan.degradeReasons.push_back(
        "runtime_base=false is not supported; forcing runtime_base=true");
  }

  const auto supported = getSupportedVendorMetrics();
  for (const auto &request : options.vendorMetrics) {
    auto metric = toLower(trim(request.name));
    if (metric == "launchstats") {
      metric = "launch_stats";
    } else if (metric == "kernel_duration_us") {
      metric = "kernel_duration";
    } else if (metric == "sm_occupancy" || metric == "nvidia_occupancy") {
      metric = "occupancy";
    } else if (metric == "sm_throughput" || metric == "nvidia_throughput") {
      metric = "throughput";
    } else if (metric == "dram_throughput" || metric == "memory_throughput" ||
               metric == "nvidia_bandwidth") {
      metric = "bandwidth";
    } else if (metric == "hardware_instruction" ||
               metric == "sm_instruction_count") {
      metric = "instruction_count";
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
    plan.degradeReasons.push_back("Unsupported NVIDIA vendor metrics: " +
                                  join(plan.disabledVendorMetrics));
  }
  const bool hardwareCounters =
      contains(plan.enabledVendorMetrics, "occupancy") ||
      contains(plan.enabledVendorMetrics, "throughput") ||
      contains(plan.enabledVendorMetrics, "bandwidth") ||
      contains(plan.enabledVendorMetrics, "instruction_count");
  if (hardwareCounters && contains(plan.enabledVendorMetrics, "instruction")) {
    // FlagPrism: CUPTI PC sampling and CUPTI KernelReplay PM counters cannot
    // share one NVIDIA profiler session. Prefer the requested hardware metric
    // and make the disabled instruction path explicit in metadata.
    plan.enabledVendorMetrics.erase(
        std::remove(plan.enabledVendorMetrics.begin(),
                    plan.enabledVendorMetrics.end(), "instruction"),
        plan.enabledVendorMetrics.end());
    plan.disabledVendorMetrics.push_back("instruction");
    plan.degradeReasons.push_back(
        "NVIDIA hardware counters take precedence over CUPTI PC sampling; "
        "instruction was disabled for this session.");
  }
  // FlagPrism: retain an explicit option in the artifact metadata so callers
  // can distinguish CUPTI vendor collection from the legacy cupti backend.
  plan.requested.adapterOptions.emplace(
      "cupti_vendor_capture",
      plan.enabledVendorMetrics.empty() ? "false" : "true");
  const bool pcSampling =
      optionEnabled(options.adapterOptions, "pcsampling") ||
      optionEnabled(options.adapterOptions, "pc_sampling") ||
      optionEnabled(options.adapterOptions, "cupti_pc_sampling");
  if (pcSampling) {
    // FlagPrism: keep the canonical option in session metadata so the shared
    // CUPTI profiler can enable PC sampling after this plan is built.
    plan.requested.adapterOptions["cupti_pc_sampling"] = "true";
  }
  plan.requested.adapterOptions["nvidia_hardware_counters"] =
      hardwareCounters ? "true" : "false";
  if (hardwareCounters) {
    // FlagPrism: the adapter configures the canonical metric name; the
    // NVPW helper maps it to the architecture-specific evaluator expression.
    std::vector<std::string> hardwareMetricNames;
    if (contains(plan.enabledVendorMetrics, "occupancy")) {
      hardwareMetricNames.push_back("occupancy");
    }
    if (contains(plan.enabledVendorMetrics, "throughput")) {
      hardwareMetricNames.push_back("throughput");
    }
    if (contains(plan.enabledVendorMetrics, "bandwidth")) {
      hardwareMetricNames.push_back("bandwidth");
    }
    if (contains(plan.enabledVendorMetrics, "instruction_count")) {
      hardwareMetricNames.push_back("instruction_count");
    }
    plan.requested.adapterOptions["nvidia_hardware_metric"] =
        join(hardwareMetricNames);
    // FlagPrism: an NVIDIA PM-counter session and the CUPTI PC-sampling
    // session are mutually exclusive; keep the effective option truthful.
    plan.requested.adapterOptions["cupti_pc_sampling"] = "false";
  }
  if (contains(plan.enabledVendorMetrics, "instruction")) {
    // FlagPrism: instruction vendor metrics use the existing CUPTI PC
    // sampling implementation and therefore share its runtime switch.
    plan.requested.adapterOptions["cupti_pc_sampling"] = "true";
  }
  return plan;
}

void NvidiaAdapter::configureRuntimeProfiler(const std::string &profilerPath,
                                             bool captureVendorEvents) const {
  // FlagPrism: the legacy cupti path is configured by Session before the
  // adapter starts the shared singleton. Preserve that behavior for packaged
  // wheels whose CUPTI library is not on the system loader path.
  auto &profiler = CuptiProfiler::instance();
  profiler.setLibPath(profilerPath);
  profiler.setHardwareCounterMetrics({});
  profiler.setPCSamplingOptional(false);
  profiler.disablePCSampling();
  profiler.enableVendorEventCapture(captureVendorEvents,
                                    /*captureMemoryActivities=*/false);
}

void NvidiaAdapter::configureRuntimeProfiler(
    const std::string &profilerPath, const VendorProfilePlan &plan) const {
  // FlagPrism: CUPTI PC sampling and activity collection share the same
  // runtime profiler. Configure both from one normalized plan so a
  // launch_stats-only session does not pay for memcpy/memset activities.
  auto &profiler = CuptiProfiler::instance();
  profiler.setLibPath(profilerPath);
  const bool pcSampling =
      optionEnabled(plan.requested.adapterOptions, "pcsampling") ||
      optionEnabled(plan.requested.adapterOptions, "cupti_pc_sampling") ||
      contains(plan.enabledVendorMetrics, "instruction");
  const bool hardwareCounters =
      contains(plan.enabledVendorMetrics, "occupancy") ||
      contains(plan.enabledVendorMetrics, "throughput") ||
      contains(plan.enabledVendorMetrics, "bandwidth") ||
      contains(plan.enabledVendorMetrics, "instruction_count");
  // FlagPrism: NVIDIA vendor PC sampling is best effort because CUPTI may be
  // disabled by the driver performance-counter permission policy.
  profiler.setPCSamplingOptional(pcSampling && !hardwareCounters);
  if (pcSampling && !hardwareCounters) {
    profiler.enablePCSampling();
  } else {
    profiler.disablePCSampling();
  }
  profiler.enableVendorEventCapture(
      !plan.enabledVendorMetrics.empty(),
      contains(plan.enabledVendorMetrics, "memory"));
  std::vector<std::string> hardwareMetricNames;
  if (contains(plan.enabledVendorMetrics, "occupancy")) {
    hardwareMetricNames.push_back("occupancy");
  }
  if (contains(plan.enabledVendorMetrics, "throughput")) {
    hardwareMetricNames.push_back("throughput");
  }
  if (contains(plan.enabledVendorMetrics, "bandwidth")) {
    hardwareMetricNames.push_back("bandwidth");
  }
  if (contains(plan.enabledVendorMetrics, "instruction_count")) {
    hardwareMetricNames.push_back("instruction_count");
  }
  profiler.setHardwareCounterMetrics(hardwareMetricNames);
}

Profiler *NvidiaAdapter::getRuntimeProfiler() const {
  // FlagPrism: capture is configured by the session plan before the shared
  // CUPTI singleton is returned, so an empty vendor metric request does not
  // retain an unbounded stream of activity keys.
  return &CuptiProfiler::instance();
}

std::unique_ptr<VendorMetricsImporter> NvidiaAdapter::createImporter() const {
  return std::make_unique<NvidiaMetricsImporter>();
}

} // namespace proton
