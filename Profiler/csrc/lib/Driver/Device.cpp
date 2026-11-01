#include "Device.h"
#ifdef FLAGPRISM_BACKEND_ENFLAME
#include "Driver/Enflame/EnflameApi.h"
#endif
// FlagPrism：query the actual MACA device properties.
#if defined(FLAGPRISM_BACKEND_METAX) || defined(FLAGPRISM_HAS_METAX)
#include <mcr/mc_runtime_api.h>
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Driver/Ascend/AscendApi.h"
#endif
// FlagPrism end
#if FLAGTREE_PROFILER_CUDA_RUNTIME
#include "Driver/GPU/CudaApi.h"
#endif
#if FLAGTREE_PROFILER_ROCTRACER_RUNTIME
#include "Driver/GPU/HipApi.h"
#endif
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Driver/Tianshu/TianshuApi.h"
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Driver/Mthreads/MthreadsApi.h"
#endif
// FlagPrism end

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  // FlagPrism：use MACA device limits and clocks, not CUDA defaults.
#if defined(FLAGPRISM_BACKEND_METAX) || defined(FLAGPRISM_HAS_METAX)
  if (type == DeviceType::METAX) {
    mcDeviceProp_t prop{};
    if (mcGetDeviceProperties(&prop, index) != mcSuccess)
      throw std::runtime_error("mcGetDeviceProperties failed");
    return Device(type, index, prop.clockRate, prop.memoryClockRate,
                  prop.memoryBusWidth, prop.multiProcessorCount, prop.name);
  }
#endif
  // FlagPrism end
#ifdef FLAGPRISM_BACKEND_ENFLAME
  if (type == DeviceType::ENFLAME) {
    return enflame::getDevice(index);
  }
#endif
#if FLAGTREE_PROFILER_CUDA_RUNTIME
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
#endif
#if FLAGTREE_PROFILER_ROCTRACER_RUNTIME
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#endif
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (type == DeviceType::ASCEND) {
    return ascend::getDevice(index);
  }
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (type == DeviceType::TIANSHU) {
    return tianshu::getDevice(index);
  }
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (type == DeviceType::MTHREADS) {
    return mthreads::getDevice(index);
  }
#endif
// FlagPrism end
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  // FlagPrism：include MetaX in portable report metadata.
  if (type == DeviceType::METAX)
    return DeviceTraits<DeviceType::METAX>::name;
  // FlagPrism end
  if (type == DeviceType::ENFLAME)
    return DeviceTraits<DeviceType::ENFLAME>::name;
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  }
  if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
  if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  }
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_METAX) && \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS)
  if (type == DeviceType::TIANSHU) {
    return DeviceTraits<DeviceType::TIANSHU>::name;
  }
#endif
// FlagPrism end
  if (type == DeviceType::MTHREADS) {
    return DeviceTraits<DeviceType::MTHREADS>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
