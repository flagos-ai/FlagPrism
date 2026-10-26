/**
 * Copyright 2025 Enflame. All Rights Reserved.
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

#ifndef __TOPSPTI_ACTIVITY_H_
#define __TOPSPTI_ACTIVITY_H_

#include <cstddef>
#include <cstdint>

#include "topspti_callbacks.h"
#include "topspti_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @defgroup Activity
 * @{
 * Functions, types, and enums that implement the TOPSPTI Activity API.
 *
 */

/**
 * \brief Flags associated with activity records.
 *
 * Activity record flags. Flags can be combined by bitwise OR to associated
 * multiple flags with an activity record. Each flag is specified to a certain
 * activity kind, as noted below.
 */
typedef enum {
  /**
   * Indicates the activity record has no flags.
   */
  TOPSPTI_ACTIVITY_FLAG_NONE = 0,

  /**
   * Indicates the activity represents an asynchronous memcpy operation. Valid
   * for TOPSPTI_ACTIVITY_KIND_MEMCPY.
   */
  TOPSPTI_ACTIVITY_FLAG_MEMCPY_ASYNC = 1 << 0,

  /**
   * When calling topsptiActivityFlushAll, this flag can be set to force TOPSPTI
   * to flush all records in the buffer, whether finished or not
   */
  TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1 << 0,

  /**
   * Indicates the activity represents an asynchronous memset operation. Valid
   * for TOPSPTI_ACTIVITY_KIND_MEMSET.
   */
  TOPSPTI_ACTIVITY_FLAG_MEMSET_ASYNC = 1 << 0,

  TOPSPTI_ACTIVITY_FLAG_FORCE_INT = 0x7fffffff
} Topspti_ActivityFlag;

/**
 * \brief The kinds of memory accessed by a memory operation/copy.
 *
 * Each kind represents the type of the memory accessed by a memory
 * operation/copy.
 */
typedef enum {
  /**
   * The memory kind is unknown.
   */
  TOPSPTI_ACTIVITY_MEMORY_KIND_UNKNOWN = 0,

  /**
   * The memory is pageable.
   */
  TOPSPTI_ACTIVITY_MEMORY_KIND_PAGEABLE = 1,

  /**
   * The memory is pinned.
   */
  TOPSPTI_ACTIVITY_MEMORY_KIND_PINNED = 2,

  /**
   * The memory is on the device.
   */
  TOPSPTI_ACTIVITY_MEMORY_KIND_DEVICE = 3,

  TOPSPTI_ACTIVITY_MEMORY_KIND_FORCE_INT = 0x7fffffff
} Topspti_ActivityMemoryKind;

/**
 * \brief The kind of a memory copy, indicating the source and destination
 * targets of the copy.
 *
 * Each kind represents the source and destination targets of a memory copy.
 * Targets are host, device, and array.
 */
typedef enum {
  /**
   * The memory copy kind is not known.
   */
  TOPSPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0,

  /**
   * A host to device memory copy.
   */
  TOPSPTI_ACTIVITY_MEMCPY_KIND_HTOD = 1,

  /**
   * A device to host memory copy.
   */
  TOPSPTI_ACTIVITY_MEMCPY_KIND_DTOH = 2,

  /**
   * A device to device memory copy on the same device.
   */
  TOPSPTI_ACTIVITY_MEMCPY_KIND_DTOD = 8,

  /**
   * A host to host memory copy.
   */
  TOPSPTI_ACTIVITY_MEMCPY_KIND_HTOH = 9,

  /**
   * A peer to peer memory copy across different devices.
   */
  TOPSPTI_ACTIVITY_MEMCPY_KIND_PTOP = 10,

  TOPSPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT = 0x7fffffff
} Topspti_ActivityMemcpyKind;

/**
 * \brief The kinds of activity records.
 *
 * Each kind of activity record represents an activity occurring on a CPU or
 * GCU. Each kind is associated with an activity record structure that holds
 * information related to that kind.
 * \see Topspti_Activity
 * \see Topspti_ActivityAPI
 * \see Topspti_ActivityMemset
 * \see Topspti_ActivityMemcpy
 * \see Topspti_ActivityKernel
 */
typedef enum {
  /**
   * The activity record is invalid.
   */
  TOPSPTI_ACTIVITY_KIND_INVALID = 0,
  /**
   * A host<->host, host<->device, or device<->device memory copy. The
   * corresponding activity record structure is \ref Topspti_ActivityMemcpy.
   */
  TOPSPTI_ACTIVITY_KIND_MEMCPY = 1,

  /**
   * A memory set executing on the GCU. The corresponding activity record
   * structure is \ref Topspti_ActivityMemset.
   */
  TOPSPTI_ACTIVITY_KIND_MEMSET = 2,

  /**
   * A kernel executing on the GCU.  The corresponding activity record structure
   * is \ref Topspti_ActivityKernel.
   */
  TOPSPTI_ACTIVITY_KIND_KERNEL = 3,

  /**
   * A tops driver API function execution. The corresponding activity record
   * structure is \ref Topspti_ActivityAPI.
   */
  TOPSPTI_ACTIVITY_KIND_DRIVER = 4,
  /**
   * A tops runtime API function execution. The corresponding activity record
   * structure is \ref Topspti_ActivityAPI.
   */
  TOPSPTI_ACTIVITY_KIND_RUNTIME = 5,

  TOPSPTI_ACTIVITY_KIND_COUNT,

  TOPSPTI_ACTIVITY_KIND_FORCE_INT = 0x7fffffff
} Topspti_ActivityKind;

/**
 * \brief The base activity record.
 *
 * The activity API uses a Topspti_Activity as a generic representation for any
 * activity. The 'kind' field is used to determine the specified activity kind,
 * and from that the Topspti_Activity object can be cast to the specified
 * activity record type appropriate for that kind.
 *
 * Note that all activity record types are padded and aligned to ensure that
 * each member of the record is naturally aligned.
 *
 * \see Topspti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT Topspti_Activity_st {
  /**
   * The kind of this activity.
   */
  Topspti_ActivityKind kind;
} Topspti_Activity;

/**
 * \brief The activity record for a runtime or runtime API invocation.
 *
 * This activity record represents an invocation of a runtime or driver API
 * (TOPSPTI_ACTIVITY_KIND_RUNTIME and TOPSPTI_ACTIVITY_KIND_DRIVER).
 */
typedef struct PACKED_ALIGNMENT Topspti_ActivityAPI_st {
  /**
   * The activity record kind, must be TOPSPTI_ACTIVITY_KIND_DRIVER or
   * TOPSPTI_ACTIVITY_KIND_RUNTIME.
   */
  Topspti_ActivityKind kind;

  /**
   * The ID of the driver or runtime function.
   */
  Topspti_CallbackId cbid;

  /**
   * The start timestamp for the function, in ns. A value of zero indicates that
   * the start time is unknown.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of zero indicates that
   * the end time is unknown.
   */
  uint64_t end;

  /**
   * ID of the process where the driver or runtime tops function is executing.
   */
  uint32_t processId;

  /**
   * ID of the thread where the driver or runtime tops function is executing.
   */
  uint32_t threadId;

  /**
   * Correlation ID of the driver or runtime tops function. Each function
   * invocation is assigned a unique correlation ID, which is equal to the
   * correlation ID in the memcpy, memset, or kernel activity record that is
   * associated with this function.
   */
  uint32_t correlationId;

  /**
   * The return value for the function. For a tops driver function with will be
   * zero, and for a tops runtime function this will be a topsError_t value.
   */
  uint32_t returnValue;
} Topspti_ActivityAPI;

/**
 * \brief The activity record for memory sets.
 *
 * This activity record represents a memory set operation
 * (TOPSPTI_ACTIVITY_KIND_MEMSET).
 */
typedef struct PACKED_ALIGNMENT Topspti_ActivityMemset_st {
  /**
   * Activity record kind, which is must be TOPSPTI_ACTIVITY_KIND_MEMSET.
   */
  Topspti_ActivityKind kind;

  /**
   * Value being assigned to memory by the memset.
   */
  uint32_t value;

  /**
   * Number of bytes being set by the memset.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memset, in ns. A value of zero indicates that
   * the start time is unknown.
   */
  uint64_t start;

  /**
   * The end timestamp for the memset, in ns. A value of zero indicates that the
   * end time is unknown.
   */
  uint64_t end;

  /**
   * ID of the device where the memset is occurring.
   */
  uint32_t deviceId;

  /**
   * ID of the context where the memset is occurring.
   */
  uint32_t contextId;

  /**
   * ID of the stream where the memset is occurring.
   */
  uint32_t streamId;

  /**
   * Correlation ID of the memset. Each memset is assigned a unique correlation
   * ID, which is equal to the correlation ID in the runtime or driver API
   * activity record that launched the memset.
   */
  uint32_t correlationId;

  /**
   * Flags associated with the memset. \see Topspti_ActivityFlag
   */
  uint16_t flags;

  /**
   * Memory kind of the memset \see Topspti_ActivityMemoryKind
   */
  uint16_t memoryKind;

  /**
   * Undefined, reserved for internal use.
   */
  uint32_t pad;

  /**
   * Undefined, reserved for internal use.
   */
  void *reserved0;
} Topspti_ActivityMemset;

/**
 * \brief The activity record for memory copies.
 *
 * This activity record represents a memory copy operation
 * (TOPSPTI_ACTIVITY_KIND_MEMCPY).
 */
typedef struct PACKED_ALIGNMENT Topspti_ActivityMemcpy_st {
  /**
   * Activity record kind, which is must be TOPSPTI_ACTIVITY_KIND_MEMCPY.
   */
  Topspti_ActivityKind kind;

  /**
   * Kind of the memcpy, stored as a byte to reduce record size.
   * \see Topspti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memcpy, stored as a byte to reduce
   * record size. \see Topspti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memcpy, stored as a byte to reduce
   * record size. \see Topspti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * Flags associated with the memcpy. \see Topspti_ActivityFlag
   */
  uint8_t flags;

  /**
   * Number of bytes transferred by the memcpy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memcpy, in ns. A value of zero indicates that
   * the start time is unknown.
   */
  uint64_t start;

  /**
   * The end timestamp for the memcpy, in ns. A value of zero indicates that the
   * end time is unknown.
   */
  uint64_t end;

  /**
   * ID of the device where the memcpy is occurring.
   */
  uint32_t deviceId;

  /**
   * ID of the context where the memcpy is occurring.
   */
  uint32_t contextId;

  /**
   * ID of the stream where the memcpy is occurring.
   */
  uint32_t streamId;

  /**
   * Correlation ID of the memcpy. Each memcpy is assigned a unique correlation
   * ID, which is equal to the correlation ID in the runtime or driver API
   * activity record that launched the memcpy.
   */
  uint32_t correlationId;

  /**
   * Undefined, reserved for internal use.
   */
  uint32_t reservedId;

  /**
   * Undefined, reserved for internal use.
   */
  uint32_t pad;

  /**
   * Undefined, reserved for internal use.
   */
  void *reserved0;
} Topspti_ActivityMemcpy;

/**
 * \brief The activity record for kernel.
 *
 * This activity record represents a kernel execution
 * (TOPSPTI_ACTIVITY_KIND_KERNEL)
 */
typedef struct PACKED_ALIGNMENT Topspti_ActivityKernel_st {
  /**
   * The activity record kind, which is must be TOPSPTI_ACTIVITY_KIND_KERNEL.
   */
  Topspti_ActivityKind kind;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t reservedSpace0;

  /**
   * The start timestamp for the kernel execution, in ns. A value of zero
   * indicates that the start time is unknown.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of zero
   * indicates that the end time is unknown.
   */
  uint64_t end;

  /**
   * The completed timestamp for the kernel execution, in ns. It represents the
   * completion of the kernel itself and all it's child kernels. A value of zero
   * indicates that the completion time is unknown.
   */
  uint64_t completed;

  /**
   * ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * Correlation ID of the kernel. Each kernel execution is assigned a unique
   * correlation ID, which is equal to the correlation ID in the runtime or
   * driver API activity record that launched the kernel.
   */
  uint32_t correlationId;

  /**
   * X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The grid ID of the kernel. Each kernel is assigned a unique grid ID at
   * runtime.
   */
  int64_t gridId;

  /**
   * The name of the kernel. This name is shared across all activity records
   * representing the same kernel, and so should not be modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} Topspti_ActivityKernel;

/**
 * \brief Enable collection of a specified kind of activity record.
 *
 * Enable collection of a specified kind of activity record. We can calling this
 * function multiple times to enable multiple kinds. All activity kinds are
 * disabled for collection by default.
 *
 * \param kind The kind of activity record to enable collection
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 * \retval TOPSPTI_ERROR_NOT_COMPATIBLE if the activity kind can not be enabled
 * \retval TOPSPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
TopsptiResult topsptiActivityEnable(Topspti_ActivityKind kind);

/**
 * \brief Disable collection of a specified kind of activity record.
 *
 * Disable collection of a specified kind of activity record. We can calling
 * this function multiple times to disable multiple kinds. All activity kinds
 * are disabled for collection by default.
 *
 * \param kind The kind of activity record to disable collection
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 * \retval TOPSPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
TopsptiResult topsptiActivityDisable(Topspti_ActivityKind kind);

/**
 * \brief Iterate over the activity records in a buffer.
 *
 * This is a helper function to iterate over the activity records in a buffer. A
 * buffer of activity records is typically obtained by receiving a
 * Topspti_BuffersCallbackCompleteFunc callback.
 *
 * An example of typical usage:
 * \code
 * Topspti_Activity *record = NULL;
 * TopsptiResult status = TOPSPTI_SUCCESS;
 *   do {
 *      status = topsptiActivityGetNextRecord(buffer, validSize, &record);
 *      if(status == TOPSPTI_SUCCESS) {
 *           // Use record here...
 *      }
 *      else if (status == TOPSPTI_ERROR_MAX_LIMIT_REACHED)
 *          break;
 *      else {
 *          goto Error;
 *      }
 *    } while (1);
 * \endcode
 *
 * \param buffer Buffer containing activity records
 * \param record Inputs the previous activity record returned by
 * topsptiActivityGetNextRecord and returns the next activity record from the
 * buffer. If input empty pointer, returns the first activity record in the
 * buffer. If one or more timestamps are not collected, records would contain
 * invalid timestamps, which is zero.
 * \param validBufferSizeBytes Number of valid bytes in the buffer.
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 * \retval TOPSPTI_ERROR_MAX_LIMIT_REACHED if no more records in the buffer
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p buffer or \p record is empty
 * pointer
 */
TopsptiResult topsptiActivityGetNextRecord(uint8_t *buffer,
                                           size_t validBufferSizeBytes,
                                           Topspti_Activity **record);

/**
 * \brief Function type for callback used by TOPSPTI to request an empty buffer
 * for storing activity records.
 *
 * This callback function signals the TOPSPTI client that an activity buffer is
 * needed by TOPSPTI. The activity buffer would be used by TOPSPTI to store
 * activity records. The callback function can decline the request by setting
 * \p *buffer to empty pointer. In this case TOPSPTI may drop activity records.
 *
 * \param buffer Returns the new buffer. If set to empty pointer then no buffer
 * is returned.
 * \param size Returns the size of the returned buffer.
 * \param maxNumRecords There is currently no actual effect. Reserved for
 * returns the max number of records that should be placed in the buffer. If set
 * as zero, the buffer would filled with as many records as possible. If greater
 * than zero, the buffer is filled with at most \p maxNumRecords records before
 * it is returned.
 */
typedef void (*Topspti_BuffersCallbackRequestFunc)(uint8_t **buffer,
                                                   size_t *size,
                                                   size_t *maxNumRecords);

/**
 * \brief Function type for callback used by TOPSPTI to deliver buffers of
 * activity records.
 *
 * This callback function deliver to the TOPSPTI client a buffer containing
 * activity records. The buffer contains \p validSize bytes of activity records
 * which can be read by topsptiActivityGetNextRecord. The number of dropped
 * records can be read by topsptiActivityGetNumDroppedRecords. TOPSPTI
 * relinquished ownership of the buffer and will not use it anymore after this
 * call. Client can deliver the buffer to TOPSPTI once more by the
 * Topspti_BuffersCallbackRequestFunc callback.
 *
 * \param buffer Activity record buffer.
 * \param size The total size of the buffer in bytes as set in
 * Topspti_BuffersCallbackRequestFunc.
 * \param validSize The size of valid bytes in the buffer.
 */
typedef void (*Topspti_BuffersCallbackCompleteFunc)(uint8_t *buffer,
                                                    size_t size,
                                                    size_t validSize);

/**
 * \brief Registers callback functions with TOPSPTI for activity buffer
 * handling.
 *
 * This function registers two callback functions, to be used in asynchronous
 * buffer handling. If registered, activity record buffers are handled using
 * asynchronous requested and completed callbacks from TOPSPTI.
 *
 * Registering the two callbacks prevents the client from using TOPSPTI's
 * blocking enqueue or dequeue functions.
 *
 * \param funcBufferRequested Callback which is invoked when an empty buffer is
 * requested by TOPSPTI
 * \param funcBufferCompleted Callback which is invoked when a buffer containing
 * activity records is delivered from TOPSPTI
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if either \p funcBufferRequested or
 * \p funcBufferCompleted is empty pointer
 */
TopsptiResult topsptiActivityRegisterCallbacks(
    Topspti_BuffersCallbackRequestFunc funcBufferRequested,
    Topspti_BuffersCallbackCompleteFunc funcBufferCompleted);

/**
 * \brief Request to deliver activity records via the buffer completion
 * callback.
 *
 * Using the callback registered in topsptiActivityRegisterCallbacks, this
 * function returns all activity records to the TOPSPTI client.
 *
 * This is a blocking call, but it doesn't issue any tops synchronization calls
 * implicitly, so it's not guaranteed that all activities are completed on the
 * underlying devices. If it has all the information filled up, including the
 * timestamps if any, activity record is considered as completed. If all
 * activity records with complete information are expected to be delivered, it
 * is the client's responsibility to issue necessary tops synchronization calls
 * before calling this function.
 *
 * Behavior of the function based on the input flag:
 * (-) ::For default flush (when flag is TOPSPTI_ACTIVITY_FLAG_NONE), it returns
 * all the activity buffers which have all the activity records completed (but
 * buffers need not to be full though). It doesn't return buffers which have
 * incomplete records. If needed, default flush can be done at a regular
 * interval in a separate thread.
 * (-) ::For forced flush (when flag is TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED), it
 * returns all the activity buffers including the buffers which have incomplete
 * activity records. It's suggested for clients to do the force flush before the
 * termination of the profiling session to allow remaining buffers to be
 * delivered. In general, it can be done in the at-exit handler.
 *
 * Before calling this function, the buffer completion callback api should be
 * activated by calling topsptiActivityRegisterCallbacks.
 *
 * \param flag The flag can be set to indicate a forced flush. See
 * Topspti_ActivityFlag
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 * \retval TOPSPTI_ERROR_INVALID_OPERATION if not preceded by a
 * successful call to topsptiActivityRegisterCallbacks
 * \retval TOPSPTI_ERROR_UNKNOWN an internal error occurred
 *
 * \see topsptiActivityFlushPeriod
 */
TopsptiResult topsptiActivityFlushAll(uint32_t flag);

/**
 * \brief Detach TOPSPTI from the running process
 *
 * This API detaches the TOPSPTI from the running process. It destroys and
 * cleans up all the resources associated with TOPSPTI in the current process.
 * After TOPSPTI detaches from the process, the process will keep on running
 * with no TOPSPTI attached to it.
 * For safe operation of the API, TOPSPTI client needs to make sure that
 * required TOPS synchronization and TOPSPTI activity buffer flush is done
 * before calling the API.
 */
TopsptiResult topsptiFinalize();

/**
 * \brief Sets the flush period for the worker thread
 *
 * TOPSPTI creates a worker thread to minimize the perturbance for the
 * application created threads. TOPSPTI offloads certain operations from the
 * application threads to the worker thread, this includes synchronization of
 * profiling resources between host and device, delivery of the activity buffers
 * to the client using the callback registered in
 * topsptiActivityRegisterCallbacks. For performance reasons, TOPSPTI wakes up
 * the worker thread based on certain heuristics.
 *
 * This API is used to control the flush period of the worker thread. This
 * setting will override the TOPSPTI heuristics. Setting time to zero disables
 * the periodic flush and restores the default behavior.
 *
 * Periodic flush can return only those activity buffers which are full and have
 * all the activity records completed.
 *
 * It's allowed to use the API \ref topsptiActivityFlushAll to flush the data
 * on-demand, even when client sets the periodic flush.
 *
 * \param time flush period in milliseconds (ms)
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 *
 * \see topsptiActivityFlushAll
 */
TopsptiResult topsptiActivityFlushPeriod(uint32_t time);

/**
 * \brief Thread-Id types.
 *
 * TOPSPTI uses different methods to obtain the thread-id depending on the
 * support and the underlying platform. This enum documents these methods for
 * each type. APIs \ref topsptiSetThreadIdType and \ref topsptiGetThreadIdType
 * can be used to set and get the thread-id type.
 */
typedef enum {
  /**
   * Default type
   * Use POSIX pthread API pthread_self()
   */
  TOPSPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT = 0,

  /**
   * This type is based on the system API available on the underlying platform
   * and thread-id obtained is supposed to be unique for the process lifetime.
   * Use syscall SYS_gettid
   */
  TOPSPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM = 1,

  /**
   * Add new enums before this field.
   */
  TOPSPTI_ACTIVITY_THREAD_ID_TYPE_SIZE = 2,

  TOPSPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT = 0x7fffffff
} Topspti_ActivityThreadIdType;

/**
 * \brief Set the thread-id type
 *
 * TOPSPTI uses the method corresponding to set type to generate the thread-id.
 * See enum \ref Topspti_ActivityThreadIdType for the list of methods.
 * Activity records having thread-id field contain the same value.
 * Thread id type must not be changed during the profiling session to
 * avoid thread-id value mismatch across activity records.
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_SUPPORTED if \p type is not supported on the
 * platform
 */
TopsptiResult topsptiSetThreadIdType(Topspti_ActivityThreadIdType type);

/**
 * \brief Get the thread-id type
 *
 * Returns the thread-id type used in TOPSPTI
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p type is empty pointer
 */
TopsptiResult topsptiGetThreadIdType(Topspti_ActivityThreadIdType *type);

/**
 * \brief Get the TOPSPTI timestamp.
 *
 * Returns a timestamp normalized to correspond with the start and end
 * timestamps reported in the TOPSPTI activity records. The timestamp is
 * reported in nanoseconds.
 *
 * \param timestamp Returns the TOPSPTI timestamp
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p timestamp is empty pointer
 */
TopsptiResult topsptiGetTimestamp(uint64_t *timestamp);

/**
 * \brief Returns the last error from a topspti call or callback
 *
 * Returns the last error that has been produced by any of the topspti api calls
 * or the callback in the same host thread and resets it to TOPSPTI_SUCCESS.
 */
TopsptiResult topsptiGetLastError();

/**
 * \brief Get the number of activity records that were dropped of insufficient
 * buffer space.
 *
 * Get the number of records that were dropped because of insufficient buffer
 * space. The dropped count includes records that could not be recorded because
 * TOPSPTI did not have activity buffer space available for the record (because
 * the Topspti_BuffersCallbackRequestFunc callback did not return an empty
 * buffer of sufficient size). The dropped count maintained for the queue is
 * reset to zero when this function is called.
 *
 * \param dropped The number of records that were dropped since the last call
 * to this function.
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p dropped is empty pointer
 */
TopsptiResult topsptiActivityGetNumDroppedRecords(size_t *dropped);

/**
 * \brief Callback function type for timestamp generation in TOPSPTI.
 *
 * This callback function would be triggered when the TOPSPTI client need obtain
 * a timestamp. The generated timestamp will serve as a normalized reference
 * time for activity tracking within TOPSPTI, including but not limited to
 * marking event start/end times in activity records. The function should return
 * a timestamp in nanoseconds.
 *
 * \see ::topsptiActivityRegisterTimestampCallback
 */
typedef uint64_t (*Topspti_TimestampCallbackFunc)(void);

/**
 * \brief Registers callback function with TOPSPTI for providing timestamp.
 *
 * TOPSPTI allows user registers a callback function to providing timestamp
 * instead of using it's default method. By default TOPSPTI uses
 * clock_gettime(CLOCK_REALTIME, ..) to retrieve the timestamp.
 *
 * To make sure all records report the timestamp using the callback function
 * registered through topsptiActivityRegisterTimestampCallback API, user should
 * done registration of timestamp callback before any of the TOPSPTI activity
 * kinds are enabled. Otherwise, part of timestamps may be missing or be report
 * through previous method.
 *
 * \param funcTimestamp Callback function which is invoked by TOPSPTI when it
 * need a timestamp.
 *
 * \retval TOPSPTI_SUCCESS
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p funcTimestamp is NULL
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED
 */
TopsptiResult topsptiActivityRegisterTimestampCallback(
    Topspti_TimestampCallbackFunc funcTimestamp);

// doxygen end group Activity
/**
 * @}
 */

#if defined(__cplusplus)
}
#endif

#endif // file guard
