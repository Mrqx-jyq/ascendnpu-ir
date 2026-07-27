#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
#define GEN_PASS_DEF_ENHANCEDFUSIONDECISION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {
struct EnhancedFusionDecisionPass
    : public hfusion::impl::EnhancedFusionDecisionBase<EnhancedFusionDecisionPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnhancedFusionDecisionPass)
  using EnhancedFusionDecisionBase::EnhancedFusionDecisionBase;
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    DenseMap<Value, Operation *> valueToProducer;
    funcOp.walk([&](Operation *op) {
      for (auto result : op->getResults()) valueToProducer[result] = op;
    });
    funcOp.walk([&](Operation *consumer) {
      for (auto operand : consumer->getOperands()) {
        auto it = valueToProducer.find(operand);
        if (it == valueToProducer.end() || it->second == consumer) continue;
        bool singleUse = true;
        for (auto result : it->second->getResults())
          if (!result.hasOneUse()) singleUse = false;
        if (singleUse && isa<linalg::LinalgOp>(it->second) && isa<linalg::LinalgOp>(consumer))
          consumer->setAttr("hfusion.fuse_with_producer", UnitAttr::get(ctx));
      }
    });
  }
};
} // namespace
std::unique_ptr<Pass> mlir::hfusion::createEnhancedFusionDecisionPass() {
  return std::make_unique<EnhancedFusionDecisionPass>();
}