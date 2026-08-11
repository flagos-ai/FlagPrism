#ifndef PROTON_DRIVER_TIANSHU_API_H_
#define PROTON_DRIVER_TIANSHU_API_H_

#include "Device.h"

#include <cstdint>

namespace proton {
namespace tianshu {

// CoreX is CUDA-driver compatible. Load it at runtime so importing the
// profiler does not require a Tianshu installation on every host.
Device getDevice(uint64_t index);

} // namespace tianshu
} // namespace proton

#endif // PROTON_DRIVER_TIANSHU_API_H_
