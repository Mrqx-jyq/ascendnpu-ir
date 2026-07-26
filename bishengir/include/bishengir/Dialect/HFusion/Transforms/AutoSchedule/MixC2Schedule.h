//===- MixC2Schedule.h - MixC2 Auto Schedule -------------------*- C++ -*-===//
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
// This file declares auto schedule policy for MixC2 kernels.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_MIXC2SCHEDULE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_MIXC2SCHEDULE_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"

namespace mlir {
namespace hfusion {

class MixC2Scheduler : public SchedulerBase {
public:
  explicit MixC2Scheduler(func::FuncOp funcOpIn)
      : SchedulerBase(funcOpIn, FusionKind::MixC2){};
  LogicalResult runOnOperation(OpBuilder &opBuilder) override;

protected:
  LogicalResult analyzeAndVerifyKernelImpl() override { return success(); }
  TilingComputeFn calculateTilingImpl() override { return nullptr; };
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override {
    return success();
  }
};

} // namespace hfusion
} // namespace mlir
#endif