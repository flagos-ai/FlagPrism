add_profiler_library(FlagTreeProfilerBindings
  Profiler.cpp
)

# FlagPrism：compile the native MACA collector with the installed SDK ABI.
if(FLAGPRISM_BACKEND STREQUAL "metax")
  find_path(FLAGPRISM_MACA_INCLUDE_DIR mcpti/mcpti.h HINTS "$ENV{MACA_PATH}/include" /opt/maca/include REQUIRED)
  find_library(FLAGPRISM_MCPTI_LIBRARY mcpti HINTS "$ENV{MACA_PATH}/lib" /opt/maca/lib REQUIRED)
  find_library(FLAGPRISM_MACA_RUNTIME mcruntime HINTS "$ENV{MACA_PATH}/lib" /opt/maca/lib REQUIRED)
  include_directories("${FLAGPRISM_MACA_INCLUDE_DIR}" "${FLAGPRISM_MACA_INCLUDE_DIR}/mcr")
  add_profiler_library(FlagTreeProfilerMetax
    lib/Profiler/Vendor/MetaxProfiler.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/Mode.cpp)
  target_link_libraries(FlagTreeProfilerMetax PUBLIC "${FLAGPRISM_MCPTI_LIBRARY}" "${FLAGPRISM_MACA_RUNTIME}")
elseif(FLAGPRISM_BACKEND STREQUAL "enflame")
# FlagPrism end
  add_profiler_library(FlagTreeProfilerEnflame
    lib/Driver/Enflame/EnflameApi.cpp
    lib/Profiler/Vendor/EnflameProfiler.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "mthreads")
  add_profiler_library(FlagTreeProfilerMthreads
    lib/Driver/Mthreads/MthreadsApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/MthreadsAdapter.cpp
    lib/Profiler/Vendor/MthreadsProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "tianshu")
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
elseif(FLAGPRISM_BACKEND STREQUAL "nvidia")
  # FlagPrism: NVIDIA vendor profiling reuses CUPTI's existing runtime
  # collector and adds only the adapter/importer layer here.
  add_profiler_library(FlagTreeProfilerNvidia
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/NvidiaAdapter.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
else()
  # FlagPrism：compile only the collector here; the shared registry is below.
  if(FLAGPRISM_HAS_METAX)
    add_profiler_library(FlagTreeProfilerMetax lib/Profiler/Vendor/MetaxProfiler.cpp)
    target_link_libraries(FlagTreeProfilerMetax PUBLIC
      "${FLAGPRISM_MCPTI_LIBRARY}" "${FLAGPRISM_MACA_RUNTIME}")
  endif()
  # FlagPrism end
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/TianshuAdapter.cpp
    lib/Profiler/Vendor/TianshuProfiler.cpp
    lib/Driver/Mthreads/MthreadsApi.cpp
    lib/Profiler/Vendor/MthreadsAdapter.cpp
    lib/Profiler/Vendor/MthreadsProfiler.cpp
    lib/Profiler/Vendor/NvidiaAdapter.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
endif()

add_subdirectory(lib)
