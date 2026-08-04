#pragma once

namespace mlir {
class DialectRegistry;
}

namespace mlir::triton::proton {

// Register Proton production passes and dialects.
void registerFlagTreeProtonPassesAndDialects(mlir::DialectRegistry &registry);

// Register the pass consumed by FlagTree's existing test/Proton lit suite.
void registerFlagTreeProtonTestPasses();

} // namespace mlir::triton::proton
