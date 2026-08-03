#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_flagtree_debugger_runtime(py::module &&m);

PYBIND11_MODULE(_native, m) {
  m.doc() = "FlagPrism debugger runtime and report component";
  init_flagtree_debugger_runtime(std::move(m));
}
