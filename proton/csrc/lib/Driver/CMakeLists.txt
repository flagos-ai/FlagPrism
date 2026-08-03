set(PROTON_DRIVER_SOURCES Device.cpp)
if(FLAGTREE_PROTON_GPU_RUNTIME)
  list(APPEND PROTON_DRIVER_SOURCES
    GPU/CudaApi.cpp
    GPU/CuptiApi.cpp
    GPU/HipApi.cpp
    GPU/HsaApi.cpp
    GPU/RoctracerApi.cpp
  )
endif()

add_proton_library(ProtonDriver ${PROTON_DRIVER_SOURCES})
