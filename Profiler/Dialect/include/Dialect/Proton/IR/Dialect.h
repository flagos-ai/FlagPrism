#ifndef DIALECT_PROTON_IR_DIALECT_H_
#define DIALECT_PROTON_IR_DIALECT_H_

#include "Profiler/Dialect/include/Dialect/Proton/IR/Dialect.h.inc"
#include "Profiler/Dialect/include/Dialect/Proton/IR/OpsEnums.h.inc"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "Profiler/Dialect/include/Dialect/Proton/IR/Ops.h.inc"

#endif // DIALECT_PROTON_IR_DIALECT_H_
