add_profiler_library(FlagTreeProfilerBindings
  Profiler.cpp
)

if(FLAGPRISM_BACKEND STREQUAL "tianshu")
  add_profiler_library(FlagTreeProfilerTianshu
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/TianshuAdapter.cpp
    lib/Profiler/Vendor/TianshuProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "ascend")
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
else()
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/TianshuAdapter.cpp
    lib/Profiler/Vendor/TianshuProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
endif()

add_subdirectory(lib)
