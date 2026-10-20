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

#ifndef __TOPSPTI_CALLBACKS_H_
#define __TOPSPTI_CALLBACKS_H_

#include <stddef.h>
#include <stdint.h>

#include "topspti_result.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @defgroup Callback
 * @{
 * Functions, types, and enums that implement the TOPSPTI Callback API.
 *
 */

/**
 * \brief Specifies the point in an API call that a callback is issued.
 *
 * Specifies the point in an API call that a callback is issued. This value is
 * communicated to the callback function via \ref
 * Topspti_CallbackData::callbackSite.
 */
typedef enum {
  /**
   * The callback is at the entry of the API call.
   */
  TOPSPTI_API_ENTER = 0,
  /**
   * The callback is at the exit of the API call.
   */
  TOPSPTI_API_EXIT = 1,
  TOPSPTI_API_CBSITE_FORCE_INT = 0x7fffffff
} Topspti_ApiCallbackSite;

/**
 * \brief Callback domains.
 *
 * Callback domains. Each domain represents callback points for a group of
 * related API functions or tops driver activity.
 */
typedef enum {
  /**
   * Invalid domain.
   */
  TOPSPTI_CB_DOMAIN_INVALID = 0,
  /**
   * Domain containing callback points for all driver API functions.
   */
  TOPSPTI_CB_DOMAIN_DRIVER_API = 1,
  /**
   * Domain containing callback points for all runtime API functions.
   */
  TOPSPTI_CB_DOMAIN_RUNTIME_API = 2,
  TOPSPTI_CB_DOMAIN_SIZE,
  TOPSPTI_CB_DOMAIN_FORCE_INT = 0x7fffffff
} Topspti_CallbackDomain;

/**
 * \brief Data passed into a driver or runtime API callback function.
 *
 * Data passed into a driver or runtime API callback function as the \p cbdata
 * argument to \ref Topspti_CallbackFunc. The \p cbdata will be this type for
 * \p domain equal to TOPSPTI_CB_DOMAIN_DRIVER_API or
 * TOPSPTI_CB_DOMAIN_RUNTIME_API. The callback data is only valid when calling
 * the callback function that passes the data. If we need to keep some data for
 * use outside of callbacks, we must copy some data. For example, if a shallow
 * copy of Toppti_CallbackData is made in a callback, we cannot dereference
 * \p functionParams outside of that callback to access function parameters.
 * \p functionName is an exception: the string pointed to by \p functionName is
 * a global constant, so it can be accessed outside of the callback.
 */
typedef struct {
  /**
   * The point in the driver or runtime function where the callback was issued.
   */
  Topspti_ApiCallbackSite callbackSite;

  /**
   * The name of the driver or runtime API function that issues the callback. It
   * is a global constant, so it can be accessed outside of the callback.
   */
  const char *functionName;

  /**
   * Pointer to the parameters passed to the driver or runtime API call. For the
   * structural definitions of parameters for each driver and runtime API
   * function, please refer to generated_tops_runtime_api_meta.h and
   * generated_tops_meta.h.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the driver or runtime API call. This field
   * is only valid in the callback of exit(TOPSPTI_API_EXIT). For a driver API
   * \p functionReturnValue points to NULL. For a runtime API
   * \p functionReturnValue points to a \p topsError_t .
   */
  void *functionReturnValue;

  /**
   * The name of the symbol operated by the driver or runtime API function that
   * issues the callback. This entry is only valid for driver and runtime launch
   * callbacks, it returns the name of the kernel.
   */
  const char *symbolName;

  /**
   * The current driver context of the thread, if there is no current context,
   * it is null. If a runtime API function initializes a context, TOPSPTI would
   * change the timing of setting this value from the entry to exit callback of
   * the runtime.
   */
  void *context;

  /**
   * The unique ID of the TOPS context associated with the thread. UIDs are
   * assigned sequentially as contexts are created and are unique within a
   * process.
   */
  uint32_t contextUid;

  /**
   * Pointer to the shared data between the entry and exit callbacks of a given
   * driver or runtime API function call. This field can be used to pass 64 bit
   * values from the entry callback to the corresponding exit callback.
   */
  uint64_t *correlationData;

  /**
   * The activity record correlation ID for this callback. For a runtime domain
   * callback (i.e. \p domain TOPSPTI_CB_DOMAIN_RUNTIME_API) this ID will equal
   * the correlation ID in the Topspti_ActivityAPI record corresponding to the
   * TOPS runtime function call. For a driver domain callback (i.e. \p domain
   * TOPSPTI_CB_DOMAIN_DRIVER_API) this ID will equal the correlation ID in the
   * Topspti_ActivityAPI record corresponding to the TOPS driver function call.
   * Within the callback, this ID can be recorded to correlate user data with
   * the activity record.
   */
  uint32_t correlationId;

} Topspti_CallbackData;

/**
 * \brief An ID for a driver API or runtime API.
 *
 * An ID for a driver API or runtime API. Within a driver API callback this
 * should be interpreted as a Topspti_driver_api_trace_cbid value (these values
 * are defined in topspti_driver_cbid.h). Within a runtime API callback this
 * should be interpreted as a Topspti_runtime_api_trace_cbid value (these values
 * are defined in topspti_runtime_cbid.h).
 */
typedef uint32_t Topspti_CallbackId;

/**
 * \brief Function type for a callback.
 *
 * Function type for a callback. The type of the data passed to the
 * callback in \p cbdata depends on the \p domain. If \p domain is
 * TOPSPTI_CB_DOMAIN_DRIVER_API or TOPSPTI_CB_DOMAIN_RUNTIME_API the type
 * of \p cbdata will be Topspti_CallbackData.
 *
 * \param userdata User data supplied at subscription of the callback
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param cbdata Data passed to the callback.
 */
typedef void (*Topspti_CallbackFunc)(void *userdata,
                                     Topspti_CallbackDomain domain,
                                     Topspti_CallbackId cbid,
                                     const void *cbdata);

/**
 * \brief A callback subscriber.
 */
typedef struct Topspti_Subscriber_st *Topspti_SubscriberHandle;

/**
 * \brief Pointer to an array of callback domains.
 */
typedef const Topspti_CallbackDomain *Topspti_DomainTable;

/**
 * \brief Get available callback domains.
 *
 * Returns in \p *domainTable an array of size \p *domainCount of all the
 * available callback domains.
 * \note \b Thread-safety: This function is thread safe.
 *
 * \param domainCount Return the number of callback domains
 * \param domainTable Return a pointer to an array of available callback domains
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p domainCount or \p domainTable
 * are NULL
 */
TopsptiResult topsptiSupportedDomains(size_t *domainCount,
                                      Topspti_DomainTable *domainTable);

/**
 * \brief Register a callback subscriber with a callback function and user data.
 *
 * Register a callback subscriber with a callback function and a pointer to user
 * data or empty. The returned subscriber handle can be used to enable or
 * disable the callback for specified domains and callback IDs.
 *
 * \note Only one subscriber can be registered status at same time point. It's
 * the responsibility of all the TOPSPTI clients to call this function before
 * starting the profiling session, to ensure that no other TOPSPTI client
 * interrupts the profiling session. This function would return the error code
 * TOPSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, if another TOPSPTI client
 * has already started a profiling session.
 * \note This function does not enable any callbacks.
 * \note \b Thread-safety: This function is thread safe.
 *
 * \param subscriber Returns handle to registered subscriber
 * \param callback The callback function
 * \param userdata A pointer to user data or empty. This data will be passed to
 * the callback function via the \p userdata parameter.
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if there is already
 * a TOPSPTI subscriber registered
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p subscriber is NULL
 */
TopsptiResult topsptiSubscribe(Topspti_SubscriberHandle *subscriber,
                               Topspti_CallbackFunc callback, void *userdata);

/**
 * \brief Unregister a callback subscriber by handle.
 *
 * Removes a callback subscriber so that no future callbacks will be issued to
 * that subscriber.
 *
 * \note \b Thread-safety: This function is thread safe.
 *
 * \param subscriber Handle to the registered subscriber
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p subscriber is NULL
 */
TopsptiResult topsptiUnsubscribe(Topspti_SubscriberHandle subscriber);

/**
 * \brief Get the current enabled/disabled state of a callback for a specified
 * domain and function ID.
 *
 * If the callback for a domain and callback ID is enabled, this function would
 * return non-zero in \p *enable , else return zero.
 *
 * \note \b Thread-safety: A subscriber must serialize access to
 * topsptiGetCallbackState, topsptiEnableCallback, topsptiEnableDomain, and
 * topsptiEnableAllDomains.
 *
 * \param enable Non-zero if callback enabled, else zero.
 * \param subscriber Handle to the registered subscriber
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p enabled is NULL, or if \p
 * subscriber, \p domain or \p cbid is unavailable.
 */
TopsptiResult topsptiGetCallbackState(uint32_t *enable,
                                      Topspti_SubscriberHandle subscriber,
                                      Topspti_CallbackDomain domain,
                                      Topspti_CallbackId cbid);

/**
 * \brief Enable or disabled callbacks for a specified domain and callback ID.
 *
 * Enable or disabled callbacks for a subscriber for a specified domain and
 * callback ID.
 *
 * \note \b Thread-safety: A subscriber must serialize access to
 * topsptiGetCallbackState, topsptiEnableCallback, topsptiEnableDomain, and
 * topsptiEnableAllDomains.
 *
 * \param enable New enable state for the callback. We set it as zero to disable
 * the callback, else to enable the callback.
 * \param subscriber - Handle for callback subscription
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p subscriber, \p domain or \p
 * cbid is unavailable.
 */
TopsptiResult topsptiEnableCallback(uint32_t enable,
                                    Topspti_SubscriberHandle subscriber,
                                    Topspti_CallbackDomain domain,
                                    Topspti_CallbackId cbid);

/**
 * \brief Enable or disable all callbacks for a specified domain.
 *
 * Enable or disable all callbacks for a specified domain.
 *
 * \note \b Thread-safety: A subscriber must serialize access to
 * topsptiGetCallbackState, topsptiEnableCallback, topsptiEnableDomain, and
 * topsptiEnableAllDomains.
 *
 * \param enable New enable state for all callbacks in the domain. We set it as
 * zero to disable all callbacks, else to enable all callbacks.
 * \param subscriber - Handle for callback subscription
 * \param domain The domain of the callback
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p subscriber or \p domain is
 * unavailable
 */
TopsptiResult topsptiEnableDomain(uint32_t enable,
                                  Topspti_SubscriberHandle subscriber,
                                  Topspti_CallbackDomain domain);

/**
 * \brief Enable or disable all callbacks in all domains.
 *
 * Enable or disable all callbacks in all domains.
 *
 * \note \b Thread-safety: A subscriber must serialize access to
 * topsptiGetCallbackState, topsptiEnableCallback, topsptiEnableDomain, and
 * topsptiEnableAllDomains.
 *
 * \param enable New enable state for all callbacks in all domain. We set it as
 * zero to disable all callbacks, else to enable all callbacks.
 * \param subscriber - Handle for callback subscription
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p subscriber is unavailable
 */
TopsptiResult topsptiEnableAllDomains(uint32_t enable,
                                      Topspti_SubscriberHandle subscriber);

/**
 * \brief Get the name of a callback for a specified domain and callback ID.
 *
 * Returns a pointer to the name c string in \p **name.
 *
 * \note \b Names are available only for the DRIVER and RUNTIME domains.
 *
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param name Returns pointer to the name string on success, NULL otherwise
 *
 * \retval TOPSPTI_SUCCESS on success
 * \retval TOPSPTI_ERROR_NOT_INITIALIZED if unable to initialize TOPSPTI
 * \retval TOPSPTI_ERROR_INVALID_PARAMETER if \p name is NULL, or if
 * \p domain or \p cbid is unavailable.
 */
TopsptiResult topsptiGetCallbackName(Topspti_CallbackDomain domain,
                                     uint32_t cbid, const char **name);

// doxygen end group Callback
/**
 * @}
 */

#if defined(__cplusplus)
}
#endif

#endif // file guard
