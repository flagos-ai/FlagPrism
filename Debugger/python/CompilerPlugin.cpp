#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_flagtree_debugger_compiler(py::module_ &m);

// FlagTree's plugin loader creates triton._C.libtriton.debugger and calls this
// symbol from the same shared object as the core IR bindings.
void init_triton_debugger(py::module &&m) {
  m.doc() = "FlagPrism debugger compiler component";
  init_flagtree_debugger_compiler(m);
}
