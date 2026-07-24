#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_flagtree_debugger_compiler(py::module_ &m);
void init_triton_debugger(py::module &&m);

PYBIND11_MODULE(_native, m) {
  m.doc() = "FlagTree debugger compiler and runtime component";
  init_flagtree_debugger_compiler(m);
  init_triton_debugger(std::move(m));
}
