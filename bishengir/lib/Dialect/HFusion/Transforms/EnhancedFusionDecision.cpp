//===- EnhancedFusionDecision.cpp - Data-reuse fusion scoring ---*- C++ -*-===//
//
// Scores producer-consumer pairs by data reuse potential.
// Annotates high-scoring pairs for downstream OpFusion.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace {

struct EnhancedFusionDecisionPass
    : public PassWrapper<EnhancedFusionDecisionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnhancedFusionDecisionPass)

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Build value -> producer mapping
    DenseMap<Value, Operation *> valueToProducer;
    funcOp.walk([&](Operation *op) {
      for (auto result : op->getResults())
        valueToProducer[result] = op;
    });

    // Score producer-consumer pairs by data reuse
    int64_t annotated = 0;
    funcOp.walk([&](Operation *consumer) {
      for (auto operand : consumer->getOperands()) {
        auto it = valueToProducer.find(operand);
        if (it == valueToProducer.end()) continue;
        auto *producer = it->second;
        if (producer == consumer) continue;

        // Score: single-use producer → high reuse (data consumed immediately)
        // Multi-use producer → lower score (data may be evicted before reuse)
        bool singleUse = true;
        for (auto result : producer->getResults())
          if (!result.hasOneUse()) singleUse = false;

        if (singleUse && isa<linalg::LinalgOp>(producer) && isa<linalg::LinalgOp>(consumer)) {
          consumer->setAttr("hfusion.fuse_with_producer", UnitAttr::get(ctx));
          ++annotated;
        }
      }
    });

    funcOp->setAttr("hfusion.fusion_annotations",
                    IntegerAttr::get(IntegerType::get(ctx, 64), annotated));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createEnhancedFusionDecisionPass(
    const EnhancedFusionDecisionOptions &options) {
  return std::make_unique<EnhancedFusionDecisionPass>();
}