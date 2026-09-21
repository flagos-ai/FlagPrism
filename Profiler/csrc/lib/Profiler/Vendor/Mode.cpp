#include "Profiler/Vendor/Mode.h"

#include "Utility/String.h"

#include <sstream>
#include <stdexcept>

namespace proton {

namespace {

bool parseBool(const std::string &value) {
  auto lower = toLower(trim(value));
  if (lower == "1" || lower == "true" || lower == "on" || lower == "yes") {
    return true;
  }
  if (lower == "0" || lower == "false" || lower == "off" || lower == "no") {
    return false;
  }
  throw std::invalid_argument("Invalid boolean value in vendor profile mode: " +
                              value);
}

std::vector<VendorMetricRequest> parseVendorMetrics(const std::string &value) {
  std::vector<VendorMetricRequest> requests;
  for (const auto &item : split(value, ",")) {
    auto metric = trim(item);
    if (metric.empty()) {
      continue;
    }
    requests.push_back(VendorMetricRequest{metric, false});
  }
  return requests;
}

} // namespace

VendorProfileOptions parseVendorProfileMode(const std::string &mode) {
  VendorProfileOptions options;
  options.rawMode = mode;

  if (trim(mode).empty()) {
    return options;
  }

  for (const auto &tokenRaw : split(mode, ":")) {
    auto token = trim(tokenRaw);
    if (token.empty()) {
      continue;
    }

    if (token == "runtime_base") {
      options.runtimeBaseEnabled = true;
      continue;
    }

    // FlagPrism: NVIDIA exposes CUPTI PC sampling as a mode token so callers
    // can request it together with vendor metrics, e.g.
    // `pcsampling:vendor_metrics=launch_stats`.
    const auto normalizedToken = toLower(token);
    if (normalizedToken == "pcsampling" || normalizedToken == "pc_sampling") {
      options.adapterOptions["pcsampling"] = "true";
      continue;
    }

    auto delimiter = token.find('=');
    if (delimiter == std::string::npos) {
      throw std::invalid_argument("Malformed vendor profile token: " + token);
    }

    auto key = trim(token.substr(0, delimiter));
    auto value = trim(token.substr(delimiter + 1));
    if (key == "runtime_base") {
      options.runtimeBaseEnabled = parseBool(value);
    } else if (key == "vendor_metrics") {
      options.vendorMetrics = parseVendorMetrics(value);
    } else {
      options.adapterOptions[key] = value;
    }
  }

  return options;
}

} // namespace proton
