//===- MixCVSchedule.cpp -- Auto-schedule fused kernels --------*- C++ -*-===//
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
// This file implements auto schedule policy for mix cv kernels.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/MixCVSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-mix-cv"
#define DBGS() (llvm::dbgs() << "[" << DEBUG_TYPE << "] [Mix CV] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// MixCVScheduler
//===----------------------------------------------------------------------===//

LogicalResult MixCVScheduler::runOnOperation(OpBuilder &opBuilder) {
  func::FuncOp mixCVFunc = getOriginalKernel();

  // Step 1: Apply LastAxisPBR opfusion (elemwise + broadcast/reduce fusion).
  // MixCV kernels consist of Cube (matmul) + Vector (elemwise) operations.
  // We first fuse all vector operations using LastAxisPBR pattern.
  HFusionOpFusionOptions options;
  options.fusionMode = FusionKind::LastAxisPBR;
  options.alwaysInline = true;
  // Fuse all tensor.empty inside and let TensorResultToOutParam do its work.
  options.moveOutToParam = false;
  FailureOr<SmallVector<func::FuncOp>> outlinedFuncs =
      applyOpFusionOutline(mixCVFunc, options);
  if (failed(outlinedFuncs))
    return mixCVFunc->emitError("Failed to apply LastAxisPBR fusion.");

  // Step 2: Apply Schedule for outlined vector kernels.
  for (auto funcOp : *outlinedFuncs) {
    LDBG("Scheduling outlined func: " << *funcOp);
    if (failed(applySchedule(funcOp, opBuilder)))
      return failure();
  }

  // Step 3: Apply TensorResultToOutParam to the original MixCV kernel.
  if (failed(applyTensorResultToOutParamsPass(mixCVFunc)))
    return failure();

  return success();
}