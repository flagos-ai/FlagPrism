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

#ifndef __TOPSPTI_RESULT_H_
#define __TOPSPTI_RESULT_H_

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @defgroup Result Codes
 * @{
 * Error and result codes returned by TOPSPTI functions.
 *
 */

/**
 * \brief TOPSPTI result codes.
 *
 * Error and result codes returned by TOPSPTI functions.
 */
typedef enum {
  /**
   * No error.
   */
  TOPSPTI_SUCCESS = 0,
  /**
   * One or more of the parameters is invalid.
   */
  TOPSPTI_ERROR_INVALID_PARAMETER = 1,
  /**
   * The current operation cannot be performed due to dependency on
   * other factors.
   */
  TOPSPTI_ERROR_INVALID_OPERATION = 7,
  /**
   * The maximum limit is reached.
   */
  TOPSPTI_ERROR_MAX_LIMIT_REACHED = 12,
  /**
   * The current operation is not compatible with the current state of the
   * object
   */
  TOPSPTI_ERROR_NOT_COMPATIBLE = 14,
  /**
   * TOPSPTI is unable to initialize its connection to the driver.
   */
  TOPSPTI_ERROR_NOT_INITIALIZED = 15,
  /**
   * Invalid kind.
   */
  TOPSPTI_ERROR_INVALID_KIND = 21,
  /**
   * The attempted operation is not supported on the current system or device.
   */
  TOPSPTI_ERROR_NOT_SUPPORTED = 27,
  /**
   * TOPSPTI doesn't allow multiple callback subscribers. Only a single
   * subscriber can be registered at a time. Same error code is used when
   * application is launched using NVIDIA tools like topsprof, Tops Visual
   * Profiler.
   */
  TOPSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED = 39,
  /**
   * An unknown internal error has occurred.
   */
  TOPSPTI_ERROR_UNKNOWN = 999,
  TOPSPTI_ERROR_FORCE_INT = 0x7fffffff
} TopsptiResult;

// doxygen end group Result Codes
/**
 * @}
 */

#if defined(__cplusplus)
}
#endif

#endif /*__TOPSPTI_RESULT_H_*/
