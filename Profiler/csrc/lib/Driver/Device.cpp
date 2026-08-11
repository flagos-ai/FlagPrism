#include "Device.h"
#if !defined(FLAGPRISM_BACKEND_TIANSHU)
#include "Driver/Ascend/AscendApi.h"
#endif
#if FLAGTREE_PROFILER_GPU_RUNTIME
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND)
#include "Driver/Tianshu/TianshuApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
#if FLAGTREE_PROFILER_GPU_RUNTIME
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#endif
#if !defined(FLAGPRISM_BACKEND_TIANSHU)
  if (type == DeviceType::ASCEND) {
    return ascend::getDevice(index);
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND)
  if (type == DeviceType::TIANSHU) {
    return tianshu::getDevice(index);
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  }
  if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
#if !defined(FLAGPRISM_BACKEND_TIANSHU)
  if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND)
  if (type == DeviceType::TIANSHU) {
    return DeviceTraits<DeviceType::TIANSHU>::name;
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
