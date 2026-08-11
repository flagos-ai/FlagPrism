#include "Profiler/Vendor/Adapter.h"

#if !defined(FLAGPRISM_BACKEND_TIANSHU)
#include "Profiler/Vendor/CannAdapter.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND)
#include "Profiler/Vendor/TianshuAdapter.h"
#endif
#include "Utility/String.h"

namespace proton {

const VendorAdapter *VendorAdapterRegistry::find(const std::string &name) {
  auto lower = toLower(name);
#if !defined(FLAGPRISM_BACKEND_TIANSHU)
  if (lower == "cann") {
    return &CannAdapter::instance();
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND)
  if (lower == "tianshu" || lower == "corex" || lower == "iluvatar") {
    return &TianshuAdapter::instance();
  }
#endif
  return nullptr;
}

std::vector<std::string> VendorAdapterRegistry::names() {
  std::vector<std::string> result;
#if !defined(FLAGPRISM_BACKEND_TIANSHU)
  result.push_back("cann");
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND)
  result.push_back("tianshu");
#endif
  return result;
}

} // namespace proton
