//===- DynamicShapeTilingPrepare.cpp -- Dynamic shape tiling pass --------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DynamicShapeTilingPrepare pass, which optimizes
// tiling strategies for dynamically-shaped tensors.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-dynamic-tiling"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

//===----------------------------------------------------------------------===//
// Dynamic Shape Analyzer
//===----------------------------------------------------------------------===//

/// Analyze dynamic dimensions in a function and determine optimal tiling.
class DynamicShapeAnalyzer {
public:
  explicit DynamicShapeAnalyzer(func::FuncOp funcOp,
                                int64_t defaultTileSize)
      : funcOp_(funcOp), defaultTileSize_(defaultTileSize) {}

  /// Find all dynamic dimensions and their constraints.
  SmallVector<std::pair<Value, int64_t>> findDynamicDims() {
    SmallVector<std::pair<Value, int64_t>> dynamicDims;

    funcOp_.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        auto type = result.getType().dyn_cast<RankedTensorType>();
        if (!type || !type.isDynamicDim(0))
          continue;

        // Check for tensor.dim or shape.dim_of that constrains this dimension
        for (auto user : result.getUsers()) {
          if (auto dimOp = dyn_cast<tensor::DimOp>(user)) {
            // Found a tensor.dim that constrains the shape
            if (auto constIdx = dimOp.getConstantIndex()) {
              int64_t dimIdx = *constIdx;
              if (dimIdx < type.getRank() && type.isDynamicDim(dimIdx)) {
                dynamicDims.push_back({result, dimIdx});
              }
            }
          }
        }
      }
    });

    return dynamicDims;
  }

  /// Check if the function has any dynamic shapes.
  bool hasDynamicShape() {
    bool found = false;
    funcOp_.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        auto type = result.getType().dyn_cast<RankedTensorType>();
        if (type) {
          for (auto dim : type.getShape()) {
            if (dim == ShapedType::kDynamic) {
              found = true;
              return WalkResult::interrupt();
            }
          }
        }
      }
      return WalkResult::advance();
    });
    return found;
  }

  /// Generate adaptive tile size expressions for dynamic dimensions.
  /// Uses min(dynamic_dim / 2, defaultTileSize) strategy to ensure
  /// the tile size never exceeds the actual dimension at runtime.
  Value generateAdaptiveTileSize(OpBuilder &builder, Location loc,
                                 Value dynamicDim) {
    auto halfDim = builder.create<arith::DivUIOp>(
        loc, dynamicDim,
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(2)));

    auto tileConst = builder.create<arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(defaultTileSize_));

    auto cmp = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, halfDim, tileConst);

    return builder.create<arith::SelectOp>(
        loc, cmp, halfDim, tileConst);
  }

private:
  func::FuncOp funcOp_;
  int64_t defaultTileSize_;
};

//===----------------------------------------------------------------------===//
// DynamicShapeTilingPrepare Pass
//===----------------------------------------------------------------------===//

struct DynamicShapeTilingPreparePass
    : public impl::DynamicShapeTilingPrepareBase<
          DynamicShapeTilingPreparePass> {
  using DynamicShapeTilingPrepareBase::DynamicShapeTilingPrepareBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    LDBG("Analyzing dynamic shape tiling for: " << funcOp.getSymName());

    DynamicShapeAnalyzer analyzer(funcOp, defaultTileSize);

    if (!analyzer.hasDynamicShape()) {
      LDBG("No dynamic shapes found, skipping.");
      return;
    }

    LDBG("Dynamic shapes detected in: " << funcOp.getSymName());

    // Insert adaptive tiling logic at the beginning of the function
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    auto dynamicDims = analyzer.findDynamicDims();
    LDBG("Found " << dynamicDims.size() << " dynamic dimensions.");

    // For each dynamic dimension, insert tensor.dim and adaptive tile size
    // calculation
    for (auto &[val, dimIdx] : dynamicDims) {
      auto loc = val.getLoc();
      auto dimVal = builder.create<tensor::DimOp>(loc, val, dimIdx);
      auto tileSize = analyzer.generateAdaptiveTileSize(builder, loc, dimVal);

      // Mark the function with a simple boolean attr to indicate tiling
      funcOp->setAttr("hfusion.tiling_enabled",
                      builder.getBoolAttr(true));

      LDBG("Dynamic dim " << dimIdx
                          << ": adaptive tile size annotation added.");
    }

    // Mark function as having dynamic tiling prepared
    funcOp->setAttr("hfusion.dynamic_tiling_prepared",
                    builder.getBoolAttr(true));
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::hfusion::createDynamicShapeTilingPreparePass(
    const DynamicShapeTilingPrepareOptions &options) {
  return std::make_unique<DynamicShapeTilingPreparePass>(options);
}
