#include "mlir/Pass/Pass.h"
#include "proton/Dialect/include/Analysis/ScopeIdAllocation.h"

using namespace mlir;
using namespace triton::proton;

namespace {

struct TestScopeIdAllocationPass
    : public PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestScopeIdAllocationPass);

  TestScopeIdAllocationPass() = default;
  TestScopeIdAllocationPass(const TestScopeIdAllocationPass &other)
      : PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>>(other) {
  }

  StringRef getArgument() const final {
    return "test-print-scope-id-allocation";
  }
  StringRef getDescription() const final {
    return "print the result of the scope id allocation pass";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleScopeIdAllocation moduleScopeIdAllocation(moduleOp);
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      mlir::emitRemark(funcOp.getLoc(), opName);
      funcOp.walk([&](RecordOp recordOp) {
        auto scopeId = moduleScopeIdAllocation.getOpScopeId(recordOp);
        mlir::emitRemark(recordOp.getLoc()) << "scope id = " << scopeId;
      });
    });
  }
};

} // namespace

namespace mlir::triton::proton {

void registerFlagTreeProtonTestPasses() {
  PassRegistration<TestScopeIdAllocationPass>();
}

} // namespace mlir::triton::proton
