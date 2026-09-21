#ifdef FLAGPRISM_BACKEND_ENFLAME
#include <tops/tops_runtime.h>
#endif
#include "Device.h"
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Driver/Ascend/AscendApi.h"
#endif
#if FLAGTREE_PROFILER_CUDA_RUNTIME
#include "Driver/GPU/CudaApi.h"
#endif
#if FLAGTREE_PROFILER_ROCTRACER_RUNTIME
#include "Driver/GPU/HipApi.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Driver/Tianshu/TianshuApi.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Driver/Mthreads/MthreadsApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
#ifdef FLAGPRISM_BACKEND_ENFLAME
  if (type == DeviceType::ENFLAME) {
    topsDeviceProp_t prop{};
    if (topsGetDeviceProperties(&prop, index) != topsSuccess)
      throw std::runtime_error("topsGetDeviceProperties failed");
    return Device(type, index, prop.clockRate, prop.memoryClockRate,
                  prop.memoryBusWidth, prop.multiProcessorCount,
                  prop.gcuArchName);
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
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (type == DeviceType::ASCEND) {
    return ascend::getDevice(index);
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (type == DeviceType::TIANSHU) {
    return tianshu::getDevice(index);
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (type == DeviceType::MTHREADS) {
    return mthreads::getDevice(index);
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::ENFLAME)
    return DeviceTraits<DeviceType::ENFLAME>::name;
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  }
  if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
  if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS)
  if (type == DeviceType::TIANSHU) {
    return DeviceTraits<DeviceType::TIANSHU>::name;
  }
#endif
  if (type == DeviceType::MTHREADS) {
    return DeviceTraits<DeviceType::MTHREADS>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
