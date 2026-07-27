#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
#define GEN_PASS_DEF_AUTOTUNINGOPTIMIZATION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {
struct AutoTuningOptimizationPass
    : public hfusion::impl::AutoTuningOptimizationBase<AutoTuningOptimizationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoTuningOptimizationPass)
  using AutoTuningOptimizationBase::AutoTuningOptimizationBase;
  void runOnOperation() override {
    auto funcOp = getOperation();
    int64_t opCount = 0;
    funcOp.walk([&](linalg::LinalgOp) { ++opCount; });
    if (opCount > 8) {
      auto ctx = &getContext();
      funcOp->setAttr("hfusion.cost_model_ops",
                      IntegerAttr::get(IntegerType::get(ctx, 64), opCount));
    }
  }
};
} // namespace
std::unique_ptr<Pass> mlir::hfusion::createAutoTuningOptimizationPass() {
  return std::make_unique<AutoTuningOptimizationPass>();
}