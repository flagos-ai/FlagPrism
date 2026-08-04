#include "Device.h"
#include "Driver/Ascend/AscendApi.h"
#if FLAGTREE_PROFILER_GPU_RUNTIME
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
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
  if (type == DeviceType::ASCEND) {
    return ascend::getDevice(index);
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  } else if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
