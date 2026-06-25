//===- SingleCubeSchedule.cpp -- Auto-schedule fused kernels -----*- C++-*-===//
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
// This file implements auto schedule policy for single cube kernels.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/SingleCubeSchedule.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <cassert>

#define DEBUG_TYPE "hfusion-single-cube-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Single Cube] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

static constexpr int64_t kL0CSizeInBytes = 128 * 1024;
static constexpr int64_t kL1SizeInBytes = 512 * 1024;

// /// Tiling Key
// static constexpr int64_t kTilingCaseKeysAttched[1] = {
//     /* kTilingCaseKeyBm128n256k256 */
//     300,
// };
/// Tiling Keys for different matrix shape categories.
/// Key 300: General purpose (balanced, M~128, N~256, K~256)
/// Key 301: Small M shapes (batch <= 128) - smaller blocks for better utilization
/// Key 302: Large matrices (M,N >= 256) - larger blocks for throughput
/// Key 303: Small N shapes (N <= 128) - narrow matrix optimization
/// Key 304: Tiny shapes (M,N < 64) - latency-sensitive, minimal blocks
static constexpr int64_t kTilingCaseKeysAttched[] = {
    300, 301, 302, 303, 304,
};

struct BlockShapeTilingData {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct ProcessShapeTilingData {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct SplitKSlicesTilingData {
  int64_t k;
};

struct SwizzleDefaultTilingData {
  int64_t direction;
  int64_t offset;
};

struct ShuffleKTypeTilingData {
  int64_t type;
};

struct EpilogueTilingData {
  int64_t pTile;
};

struct SingleCubeTilingConfig {
  BlockShapeTilingData block;
  ProcessShapeTilingData process;
  SplitKSlicesTilingData splitKSlices;
  SwizzleDefaultTilingData swizzle;
  ShuffleKTypeTilingData shuffleKType;
  EpilogueTilingData epilogue;
};

// /// Tiling Struct Default configs
// static constexpr SingleCubeTilingConfig kSingleCubeDefaultTilingInfo[1] = {
//     /* kTilingCaseKeyBm128n256k256 */
//     {{128, 256, 256}, {128, 256, 64}, {1}, {0, 3}, {0}, {4}}};

/// Tiling Struct Default configs for different matrix shape categories.
/// Format: BlockShape{M,N,K}, ProcessShape{M,N,K}, SplitK, Swizzle{dir,off}, ShuffleKType, EpilogueTile
/// L0C constraint: BlockM * BlockN * sizeof(fp16) <= 128KB (131072 bytes)
/// L1  constraint: (BlockM * BlockN + BlockK * BlockN) * 2 <= 512KB (524288 bytes)
static constexpr SingleCubeTilingConfig kSingleCubeDefaultTilingInfo[] = {
    /* Key 300: general purpose */
    {{128, 256, 256}, {128, 256, 64}, {1}, {0, 3}, {0}, {4}},
    /* Key 301: small M (batch <= 128) */
    {{64, 256, 256}, {64, 256, 64}, {1}, {0, 3}, {0}, {4}},
    /* Key 302: large square (M,N >= 256) */
    {{256, 256, 256}, {256, 256, 64}, {1}, {0, 3}, {0}, {4}},
    /* Key 303: small N (N <= 128) */
    {{128, 128, 256}, {128, 128, 128}, {1}, {0, 3}, {0}, {4}},
    /* Key 304: tiny shapes (M,N < 64) */
    {{32, 64, 128}, {32, 64, 64}, {1}, {0, 3}, {0}, {4}},
};

} // namespace

//===----------------------------------------------------------------------===//
// SingleCubeScheduler
//===----------------------------------------------------------------------===//

void buildTilingStruct(MLIRContext *ctx, const SmallVector<Expr> &exprs,
                       TilingStruct &s) {
  auto tilingDataType = IntegerType::get(ctx, 64);
  for (Expr e : exprs) {
    TilingData d = TilingData(std::move(e), tilingDataType);
    s.push_back(std::move(d));
  }
}

TilingComputeFn SingleCubeScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            StmtExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);
    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();
    TilingCases c;
    TilingStruct s;
    assert(!kernelInfo->matmulOp2Info.empty());
    auto matmulOpInfo = kernelInfo->matmulOp2Info.begin()->second;
    auto tuningInfo = kernelInfo->cubeTilingTuning;
    for (auto [tilingKey, tilingConfig] :
         llvm::zip(kTilingCaseKeysAttched, kSingleCubeDefaultTilingInfo)) {
      // Set tiling keys.
      if (failed(c.addKey(tilingKey)))
        return {};
      // Set tiling data.
      Expr lengthM = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 0);
      Expr lengthK = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 1);
      Expr lengthN = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorBId, 1);
      if (matmulOpInfo.transposeA) {
        lengthM = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 1);
        lengthK = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorAId, 0);
      }
      if (matmulOpInfo.transposeB) {
        lengthN = opBuilder->createDimSymbolExpr(matmulOpInfo.tensorBId, 0);
      }
      Expr c0 = opBuilder->createConstExpr(0);
      Expr c1 = opBuilder->createConstExpr(1);
      Expr c128 = opBuilder->createConstExpr(128);
      Expr c256 = opBuilder->createConstExpr(256);
      // Expr tilingKeyExpr = opBuilder->createConstExpr(tilingKey);
      // Expr blockTileM = opBuilder->createConstExpr(tilingConfig.block.m);

      // Shape-adaptive tiling key selection: choose best config based on M,N dims
      Expr c64 = opBuilder->createConstExpr(64);
      Expr c128_s = opBuilder->createConstExpr(128);
      Expr c256_s = opBuilder->createConstExpr(256);
      Expr c512_s = opBuilder->createConstExpr(512);

      Expr key300_v = opBuilder->createConstExpr(300LL);
      Expr key301_v = opBuilder->createConstExpr(301LL);
      Expr key302_v = opBuilder->createConstExpr(302LL);
      Expr key303_v = opBuilder->createConstExpr(303LL);
      Expr key304_v = opBuilder->createConstExpr(304LL);

      // Heuristic chain (evaluated at runtime via dynamic tiling):
      //   tiny(M<64 & N<64) -> 304, smallM(M<=128) -> 301, smallN(N<=128) -> 303,
      //   large(M>=512 & N>=512) -> 302, default -> 300
      Expr isTiny = select(lengthM < c64, lengthN < c64, c0);
      Expr isSmallM = lengthM <= c128_s;
      Expr isSmallN = lengthN <= c128_s;
      Expr isLarge = select(lengthM >= c512_s, lengthN >= c512_s, c0);

      Expr tilingKeyExpr = select(isTiny, key304_v,
                            select(isSmallM, key301_v,
                            select(isSmallN, key303_v,
                            select(isLarge, key302_v, key300_v))));
      Expr blockTileM = opBuilder->createConstExpr(tilingConfig.block.m);
      
      Expr blockTileN = opBuilder->createConstExpr(tilingConfig.block.n);
      if (!matmulOpInfo.transposeA && matmulOpInfo.transposeB) {
        blockTileM = select(lengthM <= c256, c128,
                            c128 * (lengthN <= lengthM) +
                                c256 * (c1 - (lengthN <= lengthM)));
        blockTileN = select(lengthM <= c256, c256,
                            c256 * (lengthN <= lengthM) +
                                c128 * (c1 - (lengthN <= lengthM)));
      }
      Expr blockTileK = opBuilder->createConstExpr(tilingConfig.block.k);
      if (tuningInfo.size() >= 3 && tuningInfo[0] != -1) {
        blockTileM = opBuilder->createConstExpr(tuningInfo[0]);
        blockTileN = opBuilder->createConstExpr(tuningInfo[1]);
        blockTileK = opBuilder->createConstExpr(tuningInfo[2]);
        if (tuningInfo[0] * tuningInfo[1] * sizeof(Float32Type) >
            kL0CSizeInBytes) {
          kernelInfo->originalKernel->emitError(
              "BlockM * BlockN * sizeof(float) must <= 128K(L0C Cache Size)");
        }
        if ((tuningInfo[0] * tuningInfo[1] + tuningInfo[1] * tuningInfo[2]) *
                2 >
            kL1SizeInBytes) {
          kernelInfo->originalKernel->emitError(
              "(BlockM * BlockN + BlockK * BlockN) * 2 <= 512K(L1 Cache Size)");
        }
      }
      Expr processTileM = opBuilder->createConstExpr(tilingConfig.process.m);
      Expr processTileN = opBuilder->createConstExpr(tilingConfig.process.n);
      Expr processTileK = opBuilder->createConstExpr(tilingConfig.process.k);
      Expr splitKSlices =
          opBuilder->createConstExpr(tilingConfig.splitKSlices.k);
      Expr shuffleKType =
          opBuilder->createConstExpr(tilingConfig.shuffleKType.type);
      Expr swizzleOffset =
          opBuilder->createConstExpr(tilingConfig.swizzle.offset);
      Expr swizzleDirection = select(lengthN <= lengthM, c0, c1);
      if (tuningInfo.size() >= 5 && tuningInfo[3] != -1) {
        swizzleDirection = opBuilder->createConstExpr(tuningInfo[3]);
        swizzleOffset = opBuilder->createConstExpr(tuningInfo[4]);
        if (tuningInfo[3] != 0 && tuningInfo[3] != 1) {
          kernelInfo->originalKernel->emitError(
              "swizzle direction must be one or zero!");
        }
        if (tuningInfo[4] <= 0) {
          kernelInfo->originalKernel->emitError(
              "swizzle offset must be greater than zero!");
        }
      }
      Expr epiloguePTile =
          opBuilder->createConstExpr(tilingConfig.epilogue.pTile);
      // Build tiling struct.
      SmallVector<Expr> exprs = {tilingKeyExpr, blockTileM,   blockTileN,
                                 blockTileK,    processTileM, processTileN,
                                 processTileK,  splitKSlices, swizzleDirection,
                                 swizzleOffset, shuffleKType, epiloguePTile};
      buildTilingStruct(ctx, exprs, s);
    }
    return TilingFnResultTy(std::make_pair(std::move(c), std::move(s)));
  };
}

//===----------------------------------------------------------------------===//
// Implementation of SingleCubeScheduler schedule functions.
//===----------------------------------------------------------------------===//

LogicalResult SingleCubeScheduler::createScheduleImpl(TilingKey key,
                                                      OpBuilder &opBuilder) {
#ifndef NDEBUG
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);
#endif
  return success();
}

LogicalResult
SingleCubeScheduler::runPreScheduleProcedure(OpBuilder &opBuilder) {
  func::FuncOp currentFunc = getOriginalKernel();
  // 1. apply tensor result to out params
  if (failed(applyTensorResultToOutParamsPass(currentFunc)))
    return failure();

  // 2. analyze kernel
  if (failed(analyzeAndVerifyKernel()))
    return currentFunc->emitWarning("Failed to analyze and verify kernel.");
  return success();
}

LogicalResult
SingleCubeScheduler::runPostScheduleProcedure(OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  auto tilingKey2Kernel = tilingInfo->getTilingKey2KernelMap();
  assert(!tilingKey2Kernel.empty());

  // packing tiling data to tiling struct memref
  auto tilingFunc = tilingInfo->getHostTilingFunc();
  if (failed(applyPackTilingDataPass(tilingFunc)))
    return failure();

  // mark matmul op with tiling struct memref
  BlockArgument tilingStruct = nullptr;
  auto tilingCases = tilingInfo->getTilingCases();
  func::FuncOp funcOp = tilingKey2Kernel[tilingCases[0]];
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (hacc::utils::isKernelArg(funcOp, idx,
                                 hacc::KernelArgType::kTilingStruct)) {
      tilingStruct = arg;
      break;
    }
  }
  assert(tilingStruct != nullptr);
  funcOp.walk([&](Operation *op) {
    if (isMatmulOps(op)) {
      opBuilder.setInsertionPointAfter(op);
      StringAttr tilingStructAttr = opBuilder.getStringAttr(
          stringifyEnum(hacc::KernelArgType::kTilingStruct));
      SmallVector<Attribute> arrayList{tilingStructAttr};
      ArrayAttr arrayAttr = opBuilder.getArrayAttr(arrayList);
      NamedAttribute namedAttribute(tilingStructAttr, opBuilder.getUnitAttr());
      opBuilder.create<annotation::MarkOp>(op->getLoc(), op->getResult(0),
                                           ValueRange{tilingStruct}, arrayAttr);
    }
  });
  if (failed(applyCSEAndCanonicalizePass(funcOp)))
    return failure();
  return success();
}