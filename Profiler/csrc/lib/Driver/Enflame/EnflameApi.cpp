#include "Driver/Enflame/EnflameApi.h"

#include <tops_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace proton {
namespace enflame {
namespace {

#if defined(__linux__)
// Copied verbatim from TopsPlatform 1.9.29 tops/tops_runtime_api.h.
// Scoped locally so it does not collide with the 1.10+/4.0 global typedef
// topsDeviceProp_t (= struct topsDeviceProp) from the vendored tops_runtime.h.
typedef struct topsDeviceProp_t {
  char name[256];           ///< Device name.
  size_t totalGlobalMem;    ///< Size of global memory region (in bytes).
  size_t sharedMemPerBlock; ///< Size of shared memory region (in bytes).
  int maxThreadsPerBlock;   ///< Max work items per work group or workgroup max
                            ///< size.
  int maxThreadsDim[3]; ///< Max number of threads in each dimension (XYZ) of a
                        ///< block.
  int maxGridSize[3];   ///< Max grid dimensions (XYZ).
  int clockRate;        ///< Max clock frequency of the multiProcessors in khz.
  int memoryClockRate;  ///< Max global memory clock frequency in khz.
  int memoryBusWidth;   ///< Global memory bus width in bits.
  int major;            ///< Major compute capability.
  int minor;            ///< Minor compute capability.
  int multiProcessorCount; ///< Number of multi-processors (compute units).
  int maxThreadsPerMultiProcessor; ///< Maximum resident threads per
                                   ///< multi-processor.
  int computeMode;                 ///< Compute mode.
  int clockInstructionRate; ///< Frequency in khz of the timer used by the
                            ///< device-side "clock*" instructions.
  topsDeviceArch_t arch; ///< Architectural feature flags.
  int concurrentKernels; ///< Device can possibly execute multiple kernels
                         ///< concurrently.
  int pciDomainID;       ///< PCI Domain ID
  int pciBusID;          ///< PCI Bus ID.
  int pciDeviceID;       ///< PCI Device ID.
  int pciFunctionID;     ///< PCI Function ID.A must for SRIOV
  size_t maxSharedMemoryPerMultiProcessor; ///< Maximum Shared Memory Per
                                           ///< Multiprocessor.
  int canMapHostMemory;             ///< Check whether TOPS can map host memory
  char gcuArchName[256];            ///< ENFLAME GCU Arch Name.
  int cooperativeLaunch;            ///< TOPS device supports cooperative launch
  int cooperativeMultiDeviceLaunch; ///< TOPS device supports cooperative launch
                                    ///< on multiple devices
  int kernelExecTimeoutEnabled; ///< Run time limit for kernels executed on the
                                ///< device
  int ECCEnabled;    ///< Device has ECC support enabled
  int isLargeBar;    ///< 1: if it is a large PCI bar device, else 0
  int managedMemory; ///< Device supports allocating managed memory on this
                     ///< system
  int directManagedMemAccessFromHost; ///< Host can directly access managed
                                      ///< memory on the device without
                                      ///< migration
  int concurrentManagedAccess; ///< Device can coherently access managed memory
                               ///< concurrently with the CPU
  int pageableMemoryAccess; ///< Device supports coherently accessing pageable
                            ///< memory without calling topsHostRegister on it

  int unifiedAddressing;
  // NOTE: tops extern
  int mc_channel;            ///< Memory Control Channel count.
  size_t localMemPerThread;  ///< Size of local memory region (in bytes).
  char localUuidId[16];      ///< uuid of local device
  char remoteUuidId[16][16]; ///< uuid of remote devices
  int maxSubThreadsPerBlock; ///< Max sub threads per work group size.

  int sipCount;             ///< Number of sip.
  int sicCount;             ///< Number of sic.
  int maxThreadPerSip;      ///< Max threads per sip.
  int maxSubthreadPerSip;   ///< Max sub threads per sip.
  int maxThreadPerSic;      ///< Max threads per sic.
  size_t maxLocalMemPerSip; ///< Size of local memory region pre sip (in bytes).
  size_t
      maxSharedMemPerSic; ///< Size of shared memory region pre sic (in bytes).
  int deviceMode; ///< 0: Default(means not in ASIC or DRS, can set into any of
                  ///< mode), 1: ASIC, 2: DRS.
  unsigned char remotePortId[16]; ///< The remote device's esl port ID when the
                                  ///< corresponding remoteUuidId valid.
  int deviceNodeID;               ///< device id in device node.
  int uvmSupported;
  int pciP2PSupported;
  int reserved[70]; ///< resverve for future use
} topsDeviceProp_t;

using FnGetDeviceProperties = topsError_t (*)(void *, int);
using FnRuntimeGetVersion = topsError_t (*)(int *);

constexpr int kLargeDevicePropRuntimeVersion = 1 * 1000 + 10 * 10; // 1100

struct Api {
  void *library = nullptr;
  FnGetDeviceProperties getDeviceProperties = nullptr;
  FnRuntimeGetVersion runtimeGetVersion = nullptr;
  // True for TopsPlatform >= 1.10
  bool is40Version = false;

  bool load() {
    if (library) {
      return getDeviceProperties != nullptr;
    }
    std::vector<std::string> candidates = {"libtopsrt.so.1"};
    for (const char *env : {"TOPS_HOME", "CAPS_PATH"}) {
      const char *root = std::getenv(env);
      if (root && *root) {
        candidates.push_back(std::string(root) + "/lib/libtopsrt.so.1");
      }
    }
    candidates.emplace_back("/opt/tops/lib/libtopsrt.so.1");

    for (const auto &candidate : candidates) {
      library = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (library) {
        break;
      }
    }
    if (!library) {
      return false;
    }

    getDeviceProperties = reinterpret_cast<FnGetDeviceProperties>(
        dlsym(library, "topsGetDeviceProperties"));
    runtimeGetVersion = reinterpret_cast<FnRuntimeGetVersion>(
        dlsym(library, "topsRuntimeGetVersion"));
    is40Version = detectIs40Version();
    return getDeviceProperties != nullptr;
  }

private:
  bool detectIs40Version() const {
    if (!runtimeGetVersion) {
      return false;
    }
    int runtimeVersion = 0;
    if (runtimeGetVersion(&runtimeVersion) != topsSuccess) {
      return false;
    }
    return runtimeVersion >= kLargeDevicePropRuntimeVersion;
  }
};

Api &api() {
  static Api instance;
  return instance;
}

template <typename Prop>
Device queryDevice(const Api &enflameApi, uint64_t index) {
  Prop prop{};
  if (enflameApi.getDeviceProperties(&prop, static_cast<int>(index)) !=
      topsSuccess) {
    throw std::runtime_error("topsGetDeviceProperties failed");
  }
  return Device(DeviceType::ENFLAME, index, prop.clockRate,
                prop.memoryClockRate, prop.memoryBusWidth,
                prop.multiProcessorCount, prop.gcuArchName);
}
#endif

} // namespace

Device getDevice(uint64_t index) {
#if defined(__linux__)
  auto &enflameApi = api();
  if (!enflameApi.load()) {
    throw std::runtime_error(
        "failed to load topsGetDeviceProperties from libtopsrt.so.1");
  }
  if (enflameApi.is40Version) {
    return queryDevice<topsDeviceProp>(enflameApi, index);
  }
  return queryDevice<topsDeviceProp_t>(enflameApi, index);
#else
  throw std::runtime_error("Enflame device query is only supported on Linux");
#endif
}

} // namespace enflame
} // namespace proton
