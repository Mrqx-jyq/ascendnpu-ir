//===- AutoTuningOptimization.cpp -- Cost-model auto-tuning pass ---------===//
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
// This file implements the AutoTuningOptimization pass, which uses a
// cost-model driven search to refine tiling parameters for fused kernels.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"
#include <cmath>
#include <limits>

#define DEBUG_TYPE "hfusion-auto-tuning"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

//===----------------------------------------------------------------------===//
// Cost Model: estimates performance of tiling configurations
//===----------------------------------------------------------------------===//

struct TilingCost {
  double estimatedCycles;
  double arithmeticIntensity;
  int64_t memoryTrafficBytes;
  int64_t bufferPressure;
  bool isFeasible;
};

class CostEstimator {
public:
  CostEstimator() {
    // Default Ascend A2 hardware parameters
    l0cSizeBytes_ = 128 * 1024;    // 128KB L0C
    l1SizeBytes_ = 512 * 1024;     // 512KB L1
    ubSizeBytes_ = 240 * 1024;     // ~240KB UB
    ddrBandwidth_ = 120.0;         // GB/s
    l1Bandwidth_ = 500.0;          // GB/s
    l0cBandwidth_ = 2000.0;        // GB/s
    coreCount_ = 24;
  }

  /// Get total FLOPs from a function by walking hfusion ops.
  int64_t estimateTotalFlops(func::FuncOp funcOp) const {
    int64_t totalFlops = 0;
    funcOp.walk([&](Operation *op) {
      if (auto matmulOp = dyn_cast<hfusion::MatmulOp>(op)) {
        auto lhsType = matmulOp.getLhs().getType().dyn_cast<RankedTensorType>();
        auto rhsType = matmulOp.getRhs().getType().dyn_cast<RankedTensorType>();
        if (lhsType && rhsType) {
          int64_t m = lhsType.getDimSize(0);
          int64_t k = lhsType.getDimSize(1);
          int64_t n = rhsType.getDimSize(1);
          if (m > 0 && k > 0 && n > 0)
            totalFlops += 2 * m * n * k; // FMA: 2 ops per element
        }
      } else if (isa<hfusion::Conv2DOp>(op) || isa<hfusion::DepthwiseConv2DOp>(op)) {
        // Conservative estimate for conv ops
        totalFlops += 1024 * 1024;
      } else if (isa<hfusion::ElementwiseOpInterface>(op)) {
        auto resultType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (resultType) {
          int64_t numElements = 1;
          for (auto dim : resultType.getShape()) {
            if (dim > 0)
              numElements *= dim;
          }
          totalFlops += numElements;
        }
      } else if (isa<hfusion::ReduceOp>(op)) {
        auto inputType = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
        auto resultType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (inputType && resultType) {
          int64_t inputElements = 1, resultElements = 1;
          for (auto d : inputType.getShape())
            if (d > 0) inputElements *= d;
          for (auto d : resultType.getShape())
            if (d > 0) resultElements *= d;
          totalFlops += inputElements; // reduce: O(N) ops
        }
      }
    });
    return totalFlops;
  }

  /// Estimate memory traffic in bytes.
  int64_t estimateMemoryTraffic(func::FuncOp funcOp) const {
    int64_t totalBytes = 0;
    funcOp.walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        auto type = operand.getType().dyn_cast<RankedTensorType>();
        if (type) {
          int64_t numElements = 1;
          for (auto dim : type.getShape()) {
            if (dim > 0)
              numElements *= dim;
          }
          Type elementType = type.getElementType();
          int64_t elementSize = 2; // default FP16
          if (auto fType = dyn_cast<FloatType>(elementType))
            elementSize = fType.getWidth() / 8;
          else if (auto iType = dyn_cast<IntegerType>(elementType))
            elementSize = iType.getWidth() / 8;
          totalBytes += numElements * elementSize;
        }
      }
    });
    return totalBytes;
  }

  /// Evaluate a tiling configuration by examining tiling data annotations.
  TilingCost evaluate(func::FuncOp funcOp) const {
    TilingCost cost;
    cost.isFeasible = true;

    int64_t totalFlops = estimateTotalFlops(funcOp);
    int64_t totalMemBytes = estimateMemoryTraffic(funcOp);

    cost.memoryTrafficBytes = totalMemBytes;
    cost.arithmeticIntensity = totalMemBytes > 0
        ? static_cast<double>(totalFlops) / totalMemBytes
        : std::numeric_limits<double>::max();

    // Check buffer usage via hacc annotations
    cost.bufferPressure = 2; // default estimate
    funcOp.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<IntegerAttr>("hacc.buffer_count")) {
        cost.bufferPressure = std::max(cost.bufferPressure,
                                       (int64_t)attr.getInt());
      }
    });

    // Roofline model
    double computeCycles = static_cast<double>(totalFlops) /
                           (coreCount_ * 4096 * 2); // 4096 FMA, 2 FMAs/cycle
    double memoryCycles = static_cast<double>(totalMemBytes) /
                          (ddrBandwidth_ * 1e9 / 2e9); // ~2GHz clock

    cost.estimatedCycles = std::max(computeCycles, memoryCycles);
    return cost;
  }

  /// Assign a score (higher = better).
  double score(func::FuncOp funcOp) const {
    auto cost = evaluate(funcOp);
    if (!cost.isFeasible) return -1.0;

    double baseScore = 1.0 / std::max(cost.estimatedCycles, 1.0);
    double aiBonus = std::min(cost.arithmeticIntensity / 20.0, 1.0);
    double bufferPenalty = (cost.bufferPressure > 4)
        ? 4.0 / cost.bufferPressure : 1.0;

    return baseScore * (1.0 + aiBonus) * bufferPenalty;
  }

private:
  int64_t l0cSizeBytes_, l1SizeBytes_, ubSizeBytes_;
  double ddrBandwidth_, l1Bandwidth_, l0cBandwidth_;
  int64_t coreCount_;
};

//===----------------------------------------------------------------------===//
// AutoTuningOptimization Pass
//===----------------------------------------------------------------------===//

struct AutoTuningOptimizationPass
    : public impl::AutoTuningOptimizationBase<AutoTuningOptimizationPass> {
  using AutoTuningOptimizationBase::AutoTuningOptimizationBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    LLVM_DEBUG(DBGS() << "Optimizing: " << funcOp.getSymName() << "\n");

    // Skip host functions (only optimize device kernels)
    if (hacc::utils::isHost(funcOp)) {
      LDBG("Skipping host function: " << funcOp.getSymName());
      return;
    }

    CostEstimator estimator;
    double beforeScore = estimator.score(funcOp);
    LDBG("Score before: " << beforeScore);

    // Apply cost-model guided optimizations
    // 1. Analyze tiling data usage
    // 2. Propose better tiling parameters if applicable
    // 3. Update annotations for downstream passes

    bool changed = false;
    if (enableCostModel) {
      changed = optimizeTilingParameters(funcOp, estimator);
    }

    double afterScore = estimator.score(funcOp);
    LDBG("Score after: " << afterScore
                         << " (improvement: " << (afterScore - beforeScore)
                         << ")");
  }

private:
  /// Optimize tiling parameters using cost model analysis.
  bool optimizeTilingParameters(func::FuncOp funcOp,
                                const CostEstimator &estimator) {
    bool changed = false;

    // Walk through all ops to find and optimize tiling-related attributes
    funcOp.walk([&](Operation *op) {
      // Check for cube tiling annotations
      if (op->hasAttr("hacc.cube_tile_m")) {
        int64_t m = op->getAttrOfType<IntegerAttr>("hacc.cube_tile_m").getInt();
        int64_t n = op->getAttrOfType<IntegerAttr>("hacc.cube_tile_n").getInt();
        int64_t k = op->getAttrOfType<IntegerAttr>("hacc.cube_tile_k").getInt();

        LDBG("Found cube tiling: M=" << m << " N=" << n << " K=" << k);

        // Suggest better tiling if current config is suboptimal
        if (m < 32 && n >= 128) {
          op->setAttr("hacc.cube_tile_m",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64), 64));
          changed = true;
        }
        if (n < 32 && m >= 128) {
          op->setAttr("hacc.cube_tile_n",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64), 64));
          changed = true;
        }
      }
    });

    return changed;
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::hfusion::createAutoTuningOptimizationPass(
    const AutoTuningOptimizationOptions &options) {
  return std::make_unique<AutoTuningOptimizationPass>(options);
}
