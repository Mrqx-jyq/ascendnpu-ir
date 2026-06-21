//===- MixC2Schedule.cpp -- Auto-schedule fused kernels --------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/MixC2Schedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-mix-c2"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Mix C2] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

LogicalResult MixC2Scheduler::runOnOperation(OpBuilder &opBuilder) {
  func::FuncOp mixC2Func = getOriginalKernel();

  HFusionOpFusionOptions options;
  options.fusionMode = FusionKind::LastAxisPBR;
  options.alwaysInline = true;
  options.moveOutToParam = false;
  FailureOr<SmallVector<func::FuncOp>> outlinedFuncs =
      applyOpFusionOutline(mixC2Func, options);
  if (failed(outlinedFuncs))
    return mixC2Func->emitError("Failed to apply LastAxisPBR fusion.");

  for (auto funcOp : *outlinedFuncs) {
    LDBG("Scheduling outlined func: " << *funcOp);
    if (failed(applySchedule(funcOp, opBuilder)))
      return failure();
  }

  if (failed(applyTensorResultToOutParamsPass(mixC2Func)))
    return failure();

  return success();
}
