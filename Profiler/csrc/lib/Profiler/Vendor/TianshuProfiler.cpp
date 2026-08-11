#include "Profiler/Vendor/TianshuProfiler.h"

#include "Utility/String.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace proton {
namespace {

uint64_t nowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::string normalizeColumn(std::string value) {
  std::string result;
  for (char ch : toLower(trim(value))) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      result.push_back(ch);
    }
  }
  return result;
}

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(trim(field));
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(trim(field));
  return fields;
}

std::optional<size_t> findColumn(const std::vector<std::string> &headers,
                                 std::initializer_list<const char *> names) {
  for (size_t i = 0; i < headers.size(); ++i) {
    auto normalized = normalizeColumn(headers[i]);
    for (const char *name : names) {
      if (normalized == normalizeColumn(name)) {
        return i;
      }
    }
  }
  return std::nullopt;
}

std::optional<double> parseDouble(const std::string &raw) {
  auto value = trim(raw);
  if (value.empty()) {
    return std::nullopt;
  }
  char *end = nullptr;
  errno = 0;
  double parsed = std::strtod(value.c_str(), &end);
  if (errno != 0 || end == value.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return parsed;
}

std::optional<uint64_t> parseU64(const std::string &raw) {
  auto value = trim(raw);
  if (value.empty()) {
    return std::nullopt;
  }
  char *end = nullptr;
  errno = 0;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
  if (errno != 0 || end == value.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return static_cast<uint64_t>(parsed);
}

std::optional<MetricValueType> parseMetricValue(const std::string &raw) {
  auto value = trim(raw);
  if (value.empty()) {
    return std::nullopt;
  }
  if (auto integer = parseU64(value); integer.has_value()) {
    return integer.value();
  }
  if (auto number = parseDouble(value); number.has_value()) {
    return number.value();
  }
  return value;
}

uint64_t parseTimeNs(const std::string &raw, const std::string &header) {
  auto value = parseDouble(raw);
  if (!value.has_value() || value.value() <= 0) {
    return 0;
  }
  auto normalized = normalizeColumn(header);
  if (normalized.find("ns") != std::string::npos) {
    return static_cast<uint64_t>(value.value());
  }
  if (normalized.find("ms") != std::string::npos) {
    return static_cast<uint64_t>(value.value() * 1'000'000.0);
  }
  // ixKN CSV exports commonly use microseconds for kernel timings.
  return static_cast<uint64_t>(value.value() * 1'000.0);
}

std::string cell(const std::vector<std::string> &row,
                 const std::optional<size_t> &index) {
  if (!index.has_value() || index.value() >= row.size()) {
    return {};
  }
  return trim(row[index.value()]);
}

bool looksLikeCsv(const std::filesystem::path &path) {
  return toLower(path.extension().string()) == ".csv";
}

bool looksLikeIxkn(const std::filesystem::path &path) {
  return toLower(path.extension().string()) == ".ixkn";
}

std::vector<std::filesystem::path>
collectIxknFiles(const SessionProfileMetadata &metadata) {
  std::vector<std::filesystem::path> roots;
  for (const auto &key : {"ixkn_import_path", "ixkn_output",
                          "vendor_import_path", "output_path"}) {
    auto it = metadata.config.find(key);
    if (it != metadata.config.end() && !trim(it->second).empty()) {
      roots.emplace_back(trim(it->second));
    }
  }
  if (const char *env = std::getenv("FLAGTREE_PROFILER_TIANSHU_IMPORT_PATH")) {
    if (*env) {
      roots.emplace_back(env);
    }
  }

  std::set<std::string> seen;
  std::vector<std::filesystem::path> files;
  auto add = [&](const std::filesystem::path &path) {
    auto key = path.lexically_normal().string();
    if (seen.insert(key).second) {
      files.push_back(path);
    }
  };
  for (const auto &root : roots) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
      continue;
    }
    if (std::filesystem::is_regular_file(root, ec)) {
      if (looksLikeCsv(root) || looksLikeIxkn(root)) {
        add(root);
      }
      continue;
    }
    std::filesystem::recursive_directory_iterator it(
        root, std::filesystem::directory_options::skip_permission_denied, ec);
    const std::filesystem::recursive_directory_iterator end;
    while (it != end) {
      std::error_code entryEc;
      if (it->is_regular_file(entryEc) && !entryEc &&
          (looksLikeCsv(it->path()) || looksLikeIxkn(it->path()))) {
        add(it->path());
      }
      it.increment(ec);
      if (ec) {
        ec.clear();
      }
    }
  }
  return files;
}

bool sameKernelName(const std::string &lhs, const std::string &rhs) {
  auto left = toLower(trim(lhs));
  auto right = toLower(trim(rhs));
  return !left.empty() && !right.empty() &&
         (left == right || left.find(right) != std::string::npos ||
          right.find(left) != std::string::npos);
}

void parseIxknCsv(const std::filesystem::path &file,
                  const VendorProfilePlan &plan,
                  const std::vector<RuntimeTraceEventKey> &runtimeEvents,
                  std::vector<bool> &usedRuntimeEvents,
                  VendorProfileArtifact &artifact) {
  std::ifstream input(file);
  if (!input.is_open()) {
    artifact.degradeReasons.push_back("Failed to open ixKN CSV: " +
                                     file.string());
    return;
  }
  std::string line;
  if (!std::getline(input, line)) {
    return;
  }
  auto headers = splitCsvLine(line);
  if (headers.empty()) {
    return;
  }

  const auto nameIndex = findColumn(headers, {"kernel_name", "kernelname",
                                               "op_name", "opname", "name"});
  const auto taskIndex = findColumn(headers, {"kernel_id", "task_id", "taskid"});
  const auto correlationIndex =
      findColumn(headers, {"correlation_id", "correlationid", "corrid"});
  const auto deviceIndex = findColumn(headers, {"device_id", "deviceid", "device"});
  const auto streamIndex = findColumn(headers, {"stream_id", "streamid", "stream"});
  const auto startIndex = findColumn(headers, {"start_time_ns", "starttime_ns",
                                               "start_time_us", "starttimeus",
                                               "start_time", "start"});
  const auto endIndex = findColumn(headers, {"end_time_ns", "endtime_ns",
                                             "end_time_us", "endtimeus",
                                             "end_time", "end"});
  const auto durationIndex = findColumn(
      headers, {"duration_ns", "duration_us", "duration_ms", "duration"});
  const auto sectionIndex = findColumn(headers, {"section"});
  const auto metricIndex = findColumn(headers, {"metrics", "metric"});
  const auto valueIndex =
      findColumn(headers, {"value", "metric_value", "metricvalue"});

  const bool longFormat = nameIndex.has_value() && sectionIndex.has_value() &&
                          metricIndex.has_value() && valueIndex.has_value();

  size_t parsedRows = 0;
  auto appendAssociation = [&](VendorMetricAssociation association) {
    bool matched = false;
    for (size_t i = 0; i < runtimeEvents.size(); ++i) {
      if (usedRuntimeEvents[i] ||
          !sameKernelName(association.runtimeEvent.opName,
                          runtimeEvents[i].opName)) {
        continue;
      }
      usedRuntimeEvents[i] = true;
      auto runtimeEvent = runtimeEvents[i];
      if (association.runtimeEvent.deviceId == 0) {
        association.runtimeEvent.deviceId = runtimeEvent.deviceId;
      }
      if (association.runtimeEvent.streamId == 0) {
        association.runtimeEvent.streamId = runtimeEvent.streamId;
      }
      if (association.runtimeEvent.taskId == 0) {
        association.runtimeEvent.taskId = runtimeEvent.taskId;
      }
      if (association.runtimeEvent.correlationId == 0) {
        association.runtimeEvent.correlationId = runtimeEvent.correlationId;
      }
      if (association.runtimeEvent.startTimeNs == 0) {
        association.runtimeEvent.startTimeNs = runtimeEvent.startTimeNs;
      }
      if (association.runtimeEvent.endTimeNs == 0) {
        association.runtimeEvent.endTimeNs = runtimeEvent.endTimeNs;
      }
      association.state = VendorMetricState::Collected;
      association.note = "matched ixKN row to FlagTree host runtime event";
      matched = true;
      break;
    }
    if (!matched) {
      association.state = runtimeEvents.empty() ? VendorMetricState::Collected
                                                 : VendorMetricState::Unmatched;
      association.note = runtimeEvents.empty()
                             ? "ixKN row has no FlagTree runtime event"
                             : "ixKN row could not be matched by kernel name";
    }
    artifact.associations.push_back(std::move(association));
    ++parsedRows;
  };

  if (longFormat) {
    // CoreX ixKN exports one row per kernel metric. Fold those rows into one
    // association per kernel launch before matching host runtime events.
    std::map<std::string, size_t> groupIndices;
    std::vector<VendorMetricAssociation> grouped;
    while (std::getline(input, line)) {
      auto row = splitCsvLine(line);
      if (row.empty()) {
        continue;
      }
      const auto kernelName = cell(row, nameIndex);
      const auto kernelId = cell(row, taskIndex);
      const auto context = cell(row, findColumn(headers, {"context"}));
      const auto stream = cell(row, streamIndex);
      const auto key = kernelId + "\x1f" + kernelName + "\x1f" + context +
                       "\x1f" + stream;
      auto [groupIt, inserted] = groupIndices.emplace(key, grouped.size());
      if (inserted) {
        VendorMetricAssociation association;
        association.source = "ixkn_csv";
        association.runtimeEvent.opName = kernelName;
        association.runtimeEvent.deviceId =
            parseU64(cell(row, deviceIndex)).value_or(0);
        association.runtimeEvent.streamId =
            parseU64(stream).value_or(0);
        association.runtimeEvent.taskId = parseU64(kernelId).value_or(0);
        association.metrics["ixkn_file"] = file.string();
        if (!plan.enabledVendorMetrics.empty()) {
          association.metrics["ixkn_sections"] =
              plan.requested.adapterOptions.count("ixkn_sections")
                  ? plan.requested.adapterOptions.at("ixkn_sections")
                  : std::string("configured");
        }
        grouped.push_back(std::move(association));
      }

      auto &association = grouped[groupIt->second];
      const auto metricName = normalizeColumn(cell(row, metricIndex));
      const auto sectionName = normalizeColumn(cell(row, sectionIndex));
      auto metricValue = parseMetricValue(cell(row, valueIndex));
      if (metricName.empty() || !metricValue.has_value()) {
        continue;
      }
      std::string metricKey = "tianshu." + metricName;
      if (association.metrics.count(metricKey) != 0 && !sectionName.empty()) {
        metricKey = "tianshu." + sectionName + "." + metricName;
      }
      association.metrics[metricKey] = metricValue.value();
      if (!sectionName.empty()) {
        association.metrics["tianshu.section"] = sectionName;
      }
    }
    for (auto &association : grouped) {
      appendAssociation(std::move(association));
    }
  } else {
    while (std::getline(input, line)) {
      auto row = splitCsvLine(line);
      if (row.empty()) {
        continue;
      }
      VendorMetricAssociation association;
      association.source = "ixkn_csv";
      association.runtimeEvent.opName = cell(row, nameIndex);
      association.runtimeEvent.deviceId =
          parseU64(cell(row, deviceIndex)).value_or(0);
      association.runtimeEvent.streamId =
          parseU64(cell(row, streamIndex)).value_or(0);
      association.runtimeEvent.taskId =
          parseU64(cell(row, taskIndex)).value_or(0);
      association.runtimeEvent.correlationId =
          parseU64(cell(row, correlationIndex)).value_or(0);
      association.runtimeEvent.startTimeNs = parseTimeNs(
          cell(row, startIndex), startIndex ? headers[*startIndex] : "start_us");
      association.runtimeEvent.endTimeNs = parseTimeNs(
          cell(row, endIndex), endIndex ? headers[*endIndex] : "end_us");
      if (association.runtimeEvent.endTimeNs == 0 && durationIndex) {
        auto duration =
            parseTimeNs(cell(row, durationIndex), headers[*durationIndex]);
        association.runtimeEvent.endTimeNs =
            association.runtimeEvent.startTimeNs + duration;
      }

      for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
        if (i == nameIndex || i == startIndex || i == endIndex ||
            i == durationIndex) {
          continue;
        }
        auto value = parseMetricValue(row[i]);
        if (!value.has_value()) {
          continue;
        }
        auto name = normalizeColumn(headers[i]);
        if (name.empty()) {
          continue;
        }
        association.metrics["tianshu." + name] = value.value();
      }
      association.metrics["ixkn_file"] = file.string();
      if (!plan.enabledVendorMetrics.empty()) {
        association.metrics["ixkn_sections"] =
            plan.requested.adapterOptions.count("ixkn_sections")
                ? plan.requested.adapterOptions.at("ixkn_sections")
                : std::string("configured");
      }
      appendAssociation(std::move(association));
    }
  }
  if (parsedRows == 0) {
    artifact.degradeReasons.push_back("No data rows were parsed from ixKN CSV: " +
                                     file.string());
  }
}

} // namespace

void TianshuProfiler::startOp(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  opStartTimesNs[scope.scopeId] = nowNs();
}

void TianshuProfiler::stopOp(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = opStartTimesNs.find(scope.scopeId);
  if (it == opStartTimesNs.end()) {
    return;
  }
  auto end = nowNs();
  RuntimeTraceEventKey event;
  event.scopeId = scope.scopeId;
  event.opName = scope.name;
  event.deviceId = deviceId;
  event.startTimeNs = it->second;
  event.endTimeNs = end;
  runtimeEvents.push_back(std::move(event));
  opStartTimesNs.erase(it);
}

void TianshuProfiler::doStart() {
  std::lock_guard<std::mutex> lock(mutex);
  opStartTimesNs.clear();
  runtimeEvents.clear();
  runtimeDegradeReasons.clear();
}

void TianshuProfiler::doFlush() {}

void TianshuProfiler::doStop() {
  std::lock_guard<std::mutex> lock(mutex);
  opStartTimesNs.clear();
}

void TianshuProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  std::lock_guard<std::mutex> lock(mutex);
  deviceId = 0;
  importPath.clear();
  for (const auto &raw : modeAndOptions) {
    auto token = trim(raw);
    auto separator = token.find('=');
    if (separator == std::string::npos) {
      continue;
    }
    auto key = toLower(trim(token.substr(0, separator)));
    auto value = trim(token.substr(separator + 1));
    if (key == "device_id") {
      deviceId = static_cast<uint32_t>(parseU64(value).value_or(0));
    } else if (key == "ixkn_import_path" || key == "ixkn_output" ||
               key == "vendor_import_path") {
      importPath = value;
    }
  }
}

std::vector<RuntimeTraceEventKey> TianshuProfiler::drainRuntimeEvents() {
  std::lock_guard<std::mutex> lock(mutex);
  auto events = runtimeEvents;
  runtimeEvents.clear();
  return events;
}

VendorProfileArtifact TianshuProfiler::importIxknOutput(
    const SessionProfileMetadata &metadata, const VendorProfilePlan &plan) {
  VendorProfileArtifact artifact;
  artifact.backend = metadata.backend;
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;

  auto runtimeEvents = TianshuProfiler::instance().drainRuntimeEvents();

  auto files = collectIxknFiles(metadata);
  std::vector<bool> usedRuntimeEvents(runtimeEvents.size(), false);
  for (const auto &file : files) {
    artifact.rawInputs.push_back(file.string());
    if (looksLikeCsv(file)) {
      parseIxknCsv(file, plan, runtimeEvents, usedRuntimeEvents, artifact);
    } else if (looksLikeIxkn(file)) {
      artifact.degradeReasons.push_back(
          "Retained ixKN binary export without parsing: " + file.string() +
          ". Re-run ixKN with --csv for FlagPrism vendor import.");
    }
  }

  if (files.empty() && !plan.enabledVendorMetrics.empty()) {
    artifact.degradeReasons.push_back(
        "No ixKN CSV or .ixkn export was found. Use ixkn-cli --export-profile "
        "<path> --csv and set ixkn_import_path for import.");
  }
  return artifact;
}

} // namespace proton
