// FlagPrism：explicit, single-pass MCPTI counter ranges; never infer per-kernel counters.
class MetaxCounters {
public:
  struct Range {
    uint64_t start = 0, end = 0;
    uint32_t device = 0;
    std::map<std::string, MetricValueType> metrics;
  };
  // FlagPrism：counter groups belong to the configured device context.
  int selectedDevice = -1;
  struct DeviceGuard {
    int original = -1;
    MetaxCounters &owner;
    explicit DeviceGuard(int target, MetaxCounters &owner) : owner(owner) {
      if (mcGetDevice(&original) != mcSuccess)
        throw std::runtime_error("counter device query failed");
      if (target >= 0 && mcSetDevice(target) != mcSuccess)
        throw std::runtime_error("counter device selection failed");
    }
    ~DeviceGuard() noexcept {
      if (original >= 0)
        owner.recordCleanupStatus("mcSetDevice(restore)", int(mcSetDevice(original)));
    }
  };
  // FlagPrism end
  std::vector<std::string> names;
  std::string error;
  std::vector<Range> ranges;
  void prepare() {
    error.clear(); release(); ranges.clear();
    if (names.empty()) return;
    try {
      DeviceGuard guard(selectedDevice, *this);
      int current = 0;
      if (mcGetDevice(&current) != mcSuccess)
        throw std::runtime_error("counter device query failed");
      device = current;
      mcCtx_t runtimeContext = nullptr;
      if (mcCtxGetCurrent(&runtimeContext) != mcSuccess || !runtimeContext)
        throw std::runtime_error("counter context query failed");
      auto context = reinterpret_cast<MCcontext>(runtimeContext);
      ids.clear();
      for (const auto &name : names) {
        MCpti_MetricID id;
        check(mcptiMetricGetIdFromName(device, name.c_str(), &id), "find counter metric");
        ids.push_back(id);
      }
      check(mcptiSetEventCollectionMode(context, MCPTI_EVENT_COLLECTION_MODE_CONTINUOUS),
            "set continuous counter mode");
      check(mcptiMetricCreateEventGroupSets(context, ids.size() * sizeof(ids[0]),
                                           ids.data(), &sets), "create counter groups");
      if (!sets || sets->numSets != 1)
        throw std::runtime_error("counter request requires replay passes; automatic kernel replay is unsupported");
      for (uint32_t i = 0; i < sets->sets[0].numEventGroups; ++i) {
        uint32_t all = 1;
        check(mcptiEventGroupSetAttribute(sets->sets[0].eventGroups[i],
              MCPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all),
              "enable all counter instances");
      }
    } catch (const std::exception &e) {
      error = e.what(); release();
    }
  }
  uint32_t deviceId() const { return device; }
  void begin() {
    if (!sets || active) return;
    try {
      check(mcptiEventGroupSetEnable(&sets->sets[0]), "enable counters");
      active = true;
      for (uint32_t i = 0; i < sets->sets[0].numEventGroups; ++i)
        check(mcptiEventGroupResetAllEvents(sets->sets[0].eventGroups[i]), "reset counters");
      baseline = readTotals();
      check(mcptiGetTimestamp(&start), "counter start timestamp");
    } catch (const std::exception &e) { error = e.what(); release(); }
  }
  void finish() {
    if (!sets || !active) return;
    try {
      Range range; range.device = device; range.start = start;
      check(mcptiGetTimestamp(&range.end), "counter end timestamp");
      auto totals = readTotals();
      for (auto &[id, value] : totals) {
        if (value < baseline[id])
          throw std::runtime_error("counter values are not monotonic within the active range");
        value -= baseline[id];
        range.metrics["metax.hardware.raw_event." + std::to_string(id)] = value;
      }
      std::vector<MCpti_EventID> eventIds;
      std::vector<uint64_t> values;
      for (const auto &[id, value] : totals) { eventIds.push_back(id); values.push_back(value); }
      for (size_t i = 0; i < ids.size(); ++i) {
        if (names[i] == "dram_read_throughput" || names[i] == "dram_write_throughput") {
          const auto byteName = names[i] == "dram_read_throughput" ? "dram_read_bytes" : "dram_write_bytes";
          MCpti_MetricID byteId;
          check(mcptiMetricGetIdFromName(device, byteName, &byteId), "find bandwidth byte metric");
          MCpti_MetricValue bytes{};
          check(mcptiMetricGetValue(device, byteId, eventIds.size() * sizeof(eventIds[0]),
                eventIds.data(), values.size() * sizeof(values[0]), values.data(),
                range.end - range.start, &bytes), "evaluate bandwidth bytes");
          if (range.end <= range.start) throw std::runtime_error("invalid counter range duration");
          auto key = "metax.hardware." + names[i];
          range.metrics[key] = double(bytes.metricValueUint64) * 1e9 / (range.end - range.start);
          range.metrics[key + ".bytes"] = bytes.metricValueUint64;
          range.metrics[key + ".source"] = std::string("hardware_bytes/range_duration_ns");
          continue;
        }
        MCpti_MetricValue value{};
        check(mcptiMetricGetValue(device, ids[i], eventIds.size() * sizeof(eventIds[0]),
              eventIds.data(), values.size() * sizeof(values[0]), values.data(),
              range.end - range.start, &value), "evaluate counter metric");
        MCpti_MetricValueKind kind; size_t size = sizeof(kind);
        check(mcptiMetricGetAttribute(ids[i], MCPTI_METRIC_ATTR_VALUE_KIND, &size, &kind),
              "query counter value kind");
        auto key = "metax.hardware." + names[i];
        switch (kind) {
        case MCPTI_METRIC_VALUE_KIND_DOUBLE: range.metrics[key] = value.metricValueDouble; break;
        case MCPTI_METRIC_VALUE_KIND_PERCENT: range.metrics[key] = value.metricValuePercent; break;
        case MCPTI_METRIC_VALUE_KIND_UINT64: range.metrics[key] = value.metricValueUint64; break;
        case MCPTI_METRIC_VALUE_KIND_THROUGHPUT: range.metrics[key] = value.metricValueThroughput; break;
        case MCPTI_METRIC_VALUE_KIND_INT64: range.metrics[key] = value.metricValueInt64; break;
        default: throw std::runtime_error("unsupported counter value kind");
        }
      }
      ranges.push_back(std::move(range));
      check(mcptiEventGroupSetDisable(&sets->sets[0]), "disable counters");
      active = false;
    } catch (const std::exception &e) { error = e.what(); release(); }
  }
  // FlagPrism：cleanup is best effort, but every failure remains visible in artifacts.
  void release() noexcept {
    if (sets) {
      int original = -1;
      const auto get = mcGetDevice(&original);
      const auto select = mcSetDevice(device);
      const auto disable = active ? mcptiEventGroupSetDisable(&sets->sets[0]) : MCPTI_SUCCESS;
      const auto destroy = mcptiEventGroupSetsDestroy(sets);
      const auto restore = original >= 0 ? mcSetDevice(original) : mcSuccess;
      // Attempt all cleanup operations before formatting any diagnostic.
      recordCleanupStatus("mcGetDevice(cleanup)", int(get));
      recordCleanupStatus("mcSetDevice(cleanup)", int(select));
      recordCleanupStatus("mcptiEventGroupSetDisable(cleanup)", int(disable));
      recordCleanupStatus("mcptiEventGroupSetsDestroy", int(destroy));
      recordCleanupStatus("mcSetDevice(restore)", int(restore));
    }
    active = false; sets = nullptr;
  }
  // FlagPrism end
  ~MetaxCounters() { release(); }
private:
  // FlagPrism：never throw a second exception during cleanup or destruction.
  void recordCleanupStatus(const char *operation, int status) noexcept {
    if (!status) return;
    try {
      if (!error.empty()) error += "; ";
      error += std::string(operation) + " failed: status " + std::to_string(status);
    } catch (...) {
      std::fprintf(stderr, "FlagPrism: %s failed: status %d (diagnostic allocation failed)\n",
                   operation, status);
    }
  }
  // FlagPrism end
  std::map<MCpti_EventID, uint64_t> baseline;
  std::map<MCpti_EventID, uint64_t> readTotals() {
      std::map<MCpti_EventID, uint64_t> totals;
      for (uint32_t groupIndex = 0; groupIndex < sets->sets[0].numEventGroups; ++groupIndex) {
        auto group = sets->sets[0].eventGroups[groupIndex];
        uint32_t count = groupAttribute<uint32_t>(group, MCPTI_EVENT_GROUP_ATTR_NUM_EVENTS);
        uint32_t instances = groupAttribute<uint32_t>(group, MCPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT);
        auto domain = groupAttribute<MCpti_EventDomainID>(group, MCPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID);
        uint32_t totalInstances = 0; size_t attributeSize = sizeof(totalInstances);
        check(mcptiDeviceGetEventDomainAttribute(device, domain,
              MCPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &attributeSize, &totalInstances),
              "query total counter instances");
        if (!count || !instances || !totalInstances)
          throw std::runtime_error("invalid counter instance layout");
        std::vector<MCpti_EventID> eventIds(count);
        size_t idBytes = eventIds.size() * sizeof(eventIds[0]);
        check(mcptiEventGroupGetAttribute(group, MCPTI_EVENT_GROUP_ATTR_EVENTS,
              &idBytes, eventIds.data()), "query counter events");
        for (size_t e = 0; e < count; ++e) {
          std::vector<uint64_t> values(instances);
          size_t valueBytes = values.size() * sizeof(values[0]);
          check(mcptiEventGroupReadEvent(group, MCPTI_EVENT_READ_FLAG_NONE,
                eventIds[e], &valueBytes, values.data()), "read counter event");
          if (valueBytes != values.size() * sizeof(values[0]))
            throw std::runtime_error("incomplete counter instance read");
          uint64_t sum = 0;
          for (auto value : values) {
            if (value == MCPTI_EVENT_OVERFLOW || value == MCPTI_EVENT_INVALID)
              throw std::runtime_error("invalid or overflowing counter value");
            sum += value;
          }
          totals[eventIds[e]] = sum * totalInstances / instances;
        }
      }
    return totals;
  }
  template <typename T> T groupAttribute(MCpti_EventGroup group, MCpti_EventGroupAttribute attribute) {
    T value{}; size_t size = sizeof(value);
    check(mcptiEventGroupGetAttribute(group, attribute, &size, &value), "query counter group");
    return value;
  }
  MCpti_EventGroupSets *sets = nullptr;
  std::vector<MCpti_MetricID> ids;
  uint32_t device = 0;
  uint64_t start = 0;
  bool active = false;
};
// FlagPrism end
