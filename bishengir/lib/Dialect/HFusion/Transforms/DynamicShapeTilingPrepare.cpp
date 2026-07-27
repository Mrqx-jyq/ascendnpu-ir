//===- DynamicShapeTilingPrepare.cpp - Dynamic shape tiling ----*- C++ -*-===//
//
// Inserts runtime tile size selection for dynamically-shaped tensors.
// tile = min(dim / 2, defaultTileSize)
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct DynamicShapeTilingPreparePass
    : public PassWrapper<DynamicShapeTilingPreparePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DynamicShapeTilingPreparePass)

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder builder(&getContext());
    int64_t replaced = 0;

    funcOp.walk([&](tensor::DimOp dimOp) {
      auto loc = dimOp.getLoc();
      auto dim = dimOp.getResult();

      // tile = min(dim / 2, 128)
      builder.setInsertionPointAfter(dimOp);
      auto c2 = builder.create<arith::ConstantIntOp>(loc, 2, 64);
      auto half = builder.create<arith::DivUIOp>(loc, dim, c2);
      auto c128 = builder.create<arith::ConstantIntOp>(loc, 128, 64);
      auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, half, c128);
      auto tile = builder.create<arith::SelectOp>(loc, cmp, half, c128);

      dim.replaceAllUsesExcept(tile, tile);
      ++replaced;
    });

    if (replaced > 0) {
      auto ctx = &getContext();
      funcOp->setAttr("hfusion.dyn_tile_count",
                      IntegerAttr::get(IntegerType::get(ctx, 64), replaced));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createDynamicShapeTilingPreparePass(
    const DynamicShapeTilingPrepareOptions &options) {
  return std::make_unique<DynamicShapeTilingPreparePass>();
}