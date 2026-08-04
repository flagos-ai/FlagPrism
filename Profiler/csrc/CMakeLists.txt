add_profiler_library(FlagTreeProfilerBindings
  Profiler.cpp
)

add_profiler_library(FlagTreeProfilerCann
  lib/Driver/Ascend/AscendApi.cpp
  lib/Profiler/Vendor/Adapter.cpp
  lib/Profiler/Vendor/CannAdapter.cpp
  lib/Profiler/Vendor/CannProfiler.cpp
  lib/Profiler/Vendor/Mode.cpp
)

add_subdirectory(lib)
