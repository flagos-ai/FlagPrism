/**
 * Copyright 2020-2025 The Enflame Tech Company. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Subset of the TopsPlatform runtime headers (tops/driver_types.h and
// tops/tops_runtime_api.h) used by FlagPrism. FlagPrism modules only take the
// types from here and resolve the functions from libtopsrt with dlopen; the
// declarations are for code that links libtopsrt directly, such as the
// Enflame hardware smoke test.

#ifndef FLAGPRISM_VENDORED_TOPS_RUNTIME_H
#define FLAGPRISM_VENDORED_TOPS_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum topsError {
  topsSuccess = 0,
  topsErrorInitializationError = 3,
  topsErrorSharedObjectSymbolNotFound = 302,
  topsErrorSharedObjectInitFailed = 303,
};
typedef enum topsError topsError_t;

enum topsMemcpyKind {
  topsMemcpyHostToHost = 0,
  topsMemcpyHostToDevice = 1,
  topsMemcpyDeviceToHost = 2,
  topsMemcpyDeviceToDevice = 3,
  topsMemcpyDefault = 4
};

typedef struct EfiStream_st *topsStream_t;

#define topsHostMallocDefault 0x0

typedef struct {
  unsigned feature;
} topsDeviceArch_t;

struct EfUuid_st {
  char bytes[16];
};
typedef struct EfUuid_st topsUUID_t;

// TopsPlatform >= 1.10 layout (identical in 4.0). On a 1.9 runtime the stub
// translates the older topsDeviceProp_t layout into this one.
struct topsDeviceProp {
  char name[256];
  topsUUID_t uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int multiProcessorCount;
  int integrated;
  int canMapHostMemory;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGcuBoard;
  int multiGcuBoardGroupID;
  int hostNativeAtomicSupported;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int sparseTopsArraySupported;
  int hostRegisterReadOnlySupported;
  int timelineSemaphoreInteropSupported;
  int memoryPoolsSupported;
  int gcuDirectRDMASupported;
  unsigned int gcuDirectRDMAFlushWritesOptions;
  int gcuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int deferredMappingTopsArraySupported;
  int ipcEventSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  int deviceNumaConfig;
  int deviceNumaId;
  int mpsEnabled;
  int hostNumaId;
  unsigned int gcuPciDeviceID;
  unsigned int gcuPciSubsystemID;
  int hostNumaMultinodeIpcSupported;
  int reserved[56];

  // NOTE: tops extern
  char gcuArchName[256];
  int mc_channel;
  size_t localMemPerThread;
  char localUuidId[16];
  char remoteUuidId[16][16];
  int maxSubThreadsPerBlock;
  int sipCount;
  int sicCount;
  int maxThreadPerSip;
  int maxSubthreadPerSip;
  int maxThreadPerSic;
  size_t maxLocalMemPerSip;
  size_t maxSharedMemPerSic;
  int deviceMode;
  unsigned char remotePortId[16];
  int deviceNodeID;
  int uvmSupported;
  int pciP2PSupported;
  int clockRate;
  int memoryClockRate;
  size_t maxSharedMemoryPerMultiProcessor;
  int computeMode;
  int clockInstructionRate;
  topsDeviceArch_t arch;
  int pciFunctionID;
  int isLargeBar;
  int cooperativeMultiDeviceLaunch;
  int kernelExecTimeoutEnabled;
};
typedef struct topsDeviceProp topsDeviceProp_t;

topsError_t topsGetDeviceCount(int *count);
topsError_t topsSetDevice(int device);
topsError_t topsGetDeviceProperties(topsDeviceProp_t *prop, int device);
topsError_t topsDeviceSynchronize(void);
const char *topsGetErrorString(topsError_t error);

topsError_t topsMalloc(void **devPtr, size_t size);
topsError_t topsFree(void *devPtr);
topsError_t topsHostMalloc(void **ptr, size_t size, unsigned int flags);
topsError_t topsHostFree(void *ptr);
topsError_t topsMemcpy(void *dst, const void *src, size_t count,
                       enum topsMemcpyKind kind);
topsError_t topsMemcpyAsync(void *dst, const void *src, size_t count,
                            enum topsMemcpyKind kind, topsStream_t stream);
topsError_t topsMemset(void *devPtr, int value, size_t count);
topsError_t topsMemsetAsync(void *devPtr, int value, size_t count,
                            topsStream_t stream);

topsError_t topsStreamCreate(topsStream_t *stream);
topsError_t topsStreamDestroy(topsStream_t stream);
topsError_t topsStreamSynchronize(topsStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // FLAGPRISM_VENDORED_TOPS_RUNTIME_H
