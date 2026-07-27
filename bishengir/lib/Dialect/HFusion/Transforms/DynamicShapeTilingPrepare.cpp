#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
#define GEN_PASS_DEF_DYNAMICSHAPETILINGPREPARE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {
struct DynamicShapeTilingPreparePass
    : public hfusion::impl::DynamicShapeTilingPrepareBase<DynamicShapeTilingPreparePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DynamicShapeTilingPreparePass)
  using DynamicShapeTilingPrepareBase::DynamicShapeTilingPrepareBase;
  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder builder(&getContext());
    funcOp.walk([&](tensor::DimOp dimOp) {
      auto loc = dimOp.getLoc();
      auto dim = dimOp.getResult();
      builder.setInsertionPointAfter(dimOp);
      auto c2 = builder.create<arith::ConstantIntOp>(loc, 2, 64);
      auto half = builder.create<arith::DivUIOp>(loc, dim, c2);
      auto c128 = builder.create<arith::ConstantIntOp>(loc, 128, 64);
      auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, half, c128);
      auto tile = builder.create<arith::SelectOp>(loc, cmp, half, c128);
      dim.replaceAllUsesExcept(tile, tile);
    });
  }
};
} // namespace
std::unique_ptr<Pass> mlir::hfusion::createDynamicShapeTilingPreparePass() {
  return std::make_unique<DynamicShapeTilingPreparePass>();
}