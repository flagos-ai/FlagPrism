#include "Debugger/IR/Dialect.h"
#include "Debugger/Instrumentation/Passes.h"
#include "Debugger/Metadata/Passes.h"
#include "ir.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

void loadDebuggerDialect(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::flagtree::debugger::FlagTreeDebugDialect>();
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::flagtree::debugger::FlagTreeDebugDialect>();
}

} // namespace

void init_flagtree_debugger_compiler(py::module_ &m) {
  using namespace mlir;
  using namespace mlir::flagtree::debugger;

  registerFlagTreeDebuggerMetadataPasses();
  registerFlagTreeDebuggerInstrumentationPasses();

  m.def("load_dialects", [](MLIRContext &context) {
    loadDebuggerDialect(context);
  });
  m.def("create_debug_collect_begin",
        [](TritonOpBuilder &builder, int32_t level,
           int32_t addrLevel) -> OpState {
          loadDebuggerDialect(*builder.getContext());
          auto &opBuilder = builder.getBuilder();
          auto addrLevelAttr = addrLevel < 0
                                   ? IntegerAttr()
                                   : opBuilder.getI32IntegerAttr(addrLevel);
          return builder.create<CollectBeginOp>(
              opBuilder.getI32IntegerAttr(level), addrLevelAttr,
              IntegerAttr());
        });
  m.def("create_debug_collect_end", [](TritonOpBuilder &builder) -> OpState {
    loadDebuggerDialect(*builder.getContext());
    return builder.create<CollectEndOp>(IntegerAttr());
  });
  m.def("has_debug_collect_markers",
        [](ModuleOp mod) { return hasDebugCollectMarkers(mod); });
  m.def("insert_default_debug_collect_markers",
        [](ModuleOp mod, int32_t level, int32_t addrLevel) {
          return succeeded(
              insertDefaultDebugCollectMarkers(mod, level, addrLevel));
        });
  m.def("get_debug_tracked_op_table_json",
        [](ModuleOp mod) { return getDebugTrackedOpTableJson(mod); });
  m.def("get_debug_kernel_metadata_json",
        [](ModuleOp mod) { return getDebugKernelMetadataJson(mod); });
  m.def("get_debug_kernel_id", [](ModuleOp mod) {
    return getDebugKernelId(mod);
  });
  m.def("get_debug_records_per_instance", [](ModuleOp mod) {
    return getDebugRecordsPerInstance(mod);
  });
  m.def("get_debug_record_size",
        [](ModuleOp mod) { return getDebugRecordSize(mod); });
  m.def("get_debug_record_layout",
        [](ModuleOp mod) { return getDebugRecordLayout(mod); });
  m.def("get_debug_record_plan_json",
        [](ModuleOp mod) { return getDebugRecordPlanJson(mod); });
  m.def("get_debug_full_dump_payload_bytes_per_instance", [](ModuleOp mod) {
    return getDebugFullDumpPayloadBytesPerInstance(mod);
  });
  m.def("get_debug_full_dump_plan_json",
        [](ModuleOp mod) { return getDebugFullDumpPlanJson(mod); });
  m.def("set_debug_kernel_id_seed",
        [](ModuleOp mod, const std::string &seed) {
          setDebugKernelIdSeed(mod, seed);
        });
  m.def("set_debug_hidden_arg_abi_enabled", [](ModuleOp mod, bool enabled) {
    setDebugHiddenArgAbiEnabled(mod, enabled);
  });
  m.def("set_debug_addr_level", [](ModuleOp mod, int32_t addrLevel) {
    setDebugAddrLevel(mod, addrLevel);
  });
  m.def("set_debug_timeline_enabled", [](ModuleOp mod, bool enabled) {
    setDebugTimelineEnabled(mod, enabled);
  });
  m.def("set_debug_timeline_only", [](ModuleOp mod, bool enabled) {
    setDebugTimelineOnly(mod, enabled);
  });
  m.def("assign_debug_collect_scope_ids_without_erase", [](ModuleOp mod) {
    return succeeded(assignDebugCollectScopeIdsWithoutErase(mod));
  });
  m.def("assign_debug_op_ids_and_metadata_without_pass_manager",
        [](ModuleOp mod) {
          return succeeded(
              assignDebugOpIdsAndMetadataWithoutPassManager(mod));
        });
  m.def("erase_debug_collect_markers",
        [](ModuleOp mod) { eraseDebugCollectMarkers(mod); });
  m.def("has_triton_tensor_pointer_types",
        [](ModuleOp mod) { return hasTritonTensorPointerTypes(mod); });
  m.def("add_resolve_debug_scope", [](PassManager &pm) {
    pm.addPass(createResolveDebugScopePass());
  });
  m.def("add_assign_debug_op_id", [](PassManager &pm) {
    pm.addPass(createAssignOpIdPass());
  });
  m.def("add_insert_instrumentation", [](PassManager &pm) {
    pm.addPass(createInsertInstrumentationPass());
  });
  m.def("add_simplify_record_memref_writes", [](PassManager &pm) {
    pm.addPass(createSimplifyRecordMemrefWritesPass());
  });
}
