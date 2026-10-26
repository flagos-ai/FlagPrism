#ifndef PROTON_DRIVER_ENFLAME_API_H_
#define PROTON_DRIVER_ENFLAME_API_H_

#include "Device.h"

#include <cstdint>

namespace proton {
namespace enflame {

// Query the TOPS runtime through dlopen so the profiler does not depend on a
// TopsPlatform installation at build or link time.
Device getDevice(uint64_t index);

} // namespace enflame
} // namespace proton

#endif // PROTON_DRIVER_ENFLAME_API_H_
