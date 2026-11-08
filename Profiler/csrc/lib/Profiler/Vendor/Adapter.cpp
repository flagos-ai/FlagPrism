#ifdef FLAGPRISM_BACKEND_HCU
#include "Profiler/Vendor/HcuProfiler.h"
#endif
#ifdef FLAGPRISM_BACKEND_ENFLAME
#include "Profiler/Vendor/EnflameProfiler.h"
#endif
#include "Profiler/Vendor/Adapter.h"
// FlagPrism：register native MCPTI collection.
#if defined(FLAGPRISM_BACKEND_METAX) || defined(FLAGPRISM_HAS_METAX)
#include "Profiler/Vendor/MetaxProfiler.h"
#endif
// FlagPrism end

// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Profiler/Vendor/CannAdapter.h"
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Profiler/Vendor/TianshuAdapter.h"
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Profiler/Vendor/MthreadsAdapter.h"
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
// FlagPrism: do not include the NVIDIA adapter in an Enflame-only build.
#include "Profiler/Vendor/NvidiaAdapter.h"
#endif
// FlagPrism end
#include "Utility/String.h"

namespace proton {

const VendorAdapter *VendorAdapterRegistry::find(const std::string &name) {
  auto lower = toLower(name);
#ifdef FLAGPRISM_BACKEND_HCU
  if (lower == "hcu" || lower == "hygon")
    return &HcuAdapter::instance();
#endif
  // FlagPrism：accept Triton and vendor spellings.
#if defined(FLAGPRISM_BACKEND_METAX) || defined(FLAGPRISM_HAS_METAX)
  if (lower == "metax" || lower == "maca")
    return &MetaxAdapter::instance();
#endif
  // FlagPrism end
#ifdef FLAGPRISM_BACKEND_ENFLAME
  if (lower == "enflame" || lower == "gcu" || lower == "tops")
    return &EnflameAdapter::instance();
#endif
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (lower == "cann") {
    return &CannAdapter::instance();
  }
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (lower == "tianshu" || lower == "corex" || lower == "iluvatar") {
    return &TianshuAdapter::instance();
  }
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (lower == "mthreads" || lower == "musa") {
    // FlagPrism: keep the MThreads adapter registered in mthreads-only builds.
    return &MthreadsAdapter::instance();
  }
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
  // FlagPrism: expose NVIDIA under an explicit vendor name while retaining
  // the existing "cupti" profiler entry point for compatibility.
  if (lower == "nvidia" || lower == "cuda") {
    return &NvidiaAdapter::instance();
  }
#endif
  // FlagPrism end
  return nullptr;
}

std::vector<std::string> VendorAdapterRegistry::names() {
  std::vector<std::string> result;
#ifdef FLAGPRISM_BACKEND_HCU
  result.push_back("hcu");
#endif
  // FlagPrism：publish the compiled vendor.
#if defined(FLAGPRISM_BACKEND_METAX) || defined(FLAGPRISM_HAS_METAX)
  result.push_back("metax");
#endif
  // FlagPrism end
#ifdef FLAGPRISM_BACKEND_ENFLAME
  result.push_back("enflame");
#endif
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  result.push_back("cann");
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_MTHREADS) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  result.push_back("tianshu");
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_NVIDIA)
  result.push_back("mthreads");
#endif
// FlagPrism end
// FlagPrism：exclude unrelated SDKs in MetaX builds.
#if !defined(FLAGPRISM_BACKEND_HCU) && !defined(FLAGPRISM_BACKEND_METAX) &&    \
    !defined(FLAGPRISM_BACKEND_ENFLAME) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) &&                                      \
    !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
  result.push_back("nvidia");
#endif
  // FlagPrism end
  return result;
}

} // namespace proton
