//===- AutoTuningOptimization.cpp - Cost-model tiling opt -------*- C++ -*-===//
//
// Roofline-inspired cost model: adjusts tile sizes for fused kernels
// based on op count (arithmetic intensity proxy).
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct AutoTuningOptimizationPass
    : public PassWrapper<AutoTuningOptimizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoTuningOptimizationPass)

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Count ops to estimate arithmetic intensity
    int64_t opCount = 0;
    funcOp.walk([&](linalg::LinalgOp) { ++opCount; });

    // Roofline heuristic: op-dense kernels benefit from larger tiles.
    // Annotate the function with the suggested tile adjustment.
    if (opCount > 8) {
      auto ctx = &getContext();
      funcOp->setAttr("hfusion.cost_model_ops",
                      IntegerAttr::get(IntegerType::get(ctx, 64), opCount));
      funcOp->setAttr("hfusion.cost_model_adjust",
                      IntegerAttr::get(IntegerType::get(ctx, 64),
                                       std::min(opCount / 8, (int64_t)3)));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createAutoTuningOptimizationPass(
    const AutoTuningOptimizationOptions &options) {
  return std::make_unique<AutoTuningOptimizationPass>();
}