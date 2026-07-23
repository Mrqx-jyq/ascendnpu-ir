//===- ShallowVVSchedule.cpp -- Auto-schedule fused kernels -----*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ShallowVVSchedule.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-shallow-vv"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Shallow VV] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// ShallowVVScheduler
//===----------------------------------------------------------------------===//

LogicalResult ShallowVVScheduler::runOnOperation(OpBuilder &opBuilder) {
  func::FuncOp shallowVVFunc = getOriginalKernel();

  // Step 1: Apply PureElemwise opfusion within the ShallowVV kernel.
  HFusionOpFusionOptions options;
  options.fusionMode = FusionKind::PureElemwise;
  options.alwaysInline = true;
  options.moveOutToParam = false;
  FailureOr<SmallVector<func::FuncOp>> outlinedFuncs =
      applyOpFusionOutline(shallowVVFunc, options);
  if (failed(outlinedFuncs))
    return shallowVVFunc->emitError("Failed to apply PureElemwise fusion.");

  // Step 2: Apply Schedule for each outlined kernel.
  for (auto funcOp : *outlinedFuncs) {
    LDBG("Scheduling outlined func: " << *funcOp);
    if (failed(applySchedule(funcOp, opBuilder)))
      return failure();
  }

  // Step 3: Apply TensorResultToOutParam to the original ShallowVV kernel.
  if (failed(applyTensorResultToOutParamsPass(shallowVVFunc)))
    return failure();

  return success();
}
