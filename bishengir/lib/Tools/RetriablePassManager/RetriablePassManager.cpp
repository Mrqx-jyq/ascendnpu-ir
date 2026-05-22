//===- RetriablePassManager.cpp - Retriable pass manager --------*- C++ -*-===//
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

#include "bishengir/Tools/RetriablePassManager/RetriablePassManager.h"

#include "bishengir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bishengir-retriable-pass-manager"

using namespace mlir;
using namespace bishengir;

RetriablePassManager::RetriablePassManager(BiShengIRCompileMainConfig &config,
                                           MLIRContext *ctx)
    : config(config), context(ctx) {}

void RetriablePassManager::addPolicy(std::unique_ptr<RetryPolicy> policy) {
  policies.push_back(std::move(policy));
}

void RetriablePassManager::emitFallbackNote(llvm::StringRef retryCause,
                                            llvm::StringRef disabledOption) {
  llvm::errs() << "[NOTE] " << retryCause
               << " detected; automatically disabled " << disabledOption
               << " and retrying compilation.\n";
}

void RetriablePassManager::emitFallbackSummary(
    llvm::ArrayRef<AppliedCompileFallback> appliedFallbacks,
    bool compilationSucceeded) {
  if (appliedFallbacks.empty())
    return;

  const std::string &firstCause = appliedFallbacks.front().retryCause;
  const bool allSameCause = llvm::all_of(appliedFallbacks, [&](const AppliedCompileFallback &fb) {
    return fb.retryCause == firstCause;
  });

  llvm::errs() << "[NOTE] ";
  if (allSameCause) {
    llvm::errs() << "Due to " << firstCause << ", the following option"
                 << (appliedFallbacks.size() == 1 ? " was" : "s were")
                 << " automatically disabled: ";
    llvm::SmallVector<llvm::StringRef, 8> optionNames;
    for (const auto &fb : appliedFallbacks)
      optionNames.push_back(fb.disabledOption);
    llvm::interleaveComma(optionNames, llvm::errs());
  } else {
    llvm::errs() << "The following options were automatically disabled: ";
    for (size_t i = 0, e = appliedFallbacks.size(); i != e; ++i) {
      if (i != 0)
        llvm::errs() << ", ";
      llvm::errs() << appliedFallbacks[i].disabledOption << " (due to "
                   << appliedFallbacks[i].retryCause << ")";
    }
  }
  if (compilationSucceeded)
    llvm::errs() << "; compilation then succeeded.\n";
  else
    llvm::errs() << "; compilation still failed.\n";
}

LogicalResult RetriablePassManager::runOnce(ModuleOp mod,
                                            const BuildPipelineFn &buildPipeline,
                                            StringRef pipelineName) {
  BiShengIRPassManager passManager(config, context,
                                   ModuleOp::getOperationName(),
                                   OpPassManager::Nesting::Implicit);
  buildPipeline(passManager);

  (void)mlir::applyPassManagerCLOptions(passManager);
  (void)bishengir::applyPassManagerCLOptions(passManager);

  if (failed(passManager.run(mod)))
    return mod->emitError("Failed to run " + pipelineName.str() + " pipeline\n");

  return success();
}

LogicalResult RetriablePassManager::runWithRetry(
    ModuleOp &mod, const BuildPipelineFn &buildPipeline, StringRef pipelineName,
    std::vector<std::unique_ptr<Diagnostic>> &collectedDiagnostics,
    std::vector<AppliedCompileFallback> &appliedFallbacks) {
  while (true) {
    for (const std::unique_ptr<RetryPolicy> &policy : policies)
      policy->onBeforePipelineAttempt();

    const size_t pipelineAttemptDiagStart = collectedDiagnostics.size();
    OwningOpRef<ModuleOp> attempt = cast<ModuleOp>(mod->clone());
    ModuleOp attemptModule = attempt.get();

    if (succeeded(runOnce(attemptModule, buildPipeline, pipelineName))) {
      collectedDiagnostics.resize(pipelineAttemptDiagStart);
      mod.erase();
      mod = attemptModule;
      attempt.release();
      return success();
    }

    if (policies.empty())
      return failure();

    llvm::ArrayRef<std::unique_ptr<Diagnostic>> attemptDiagnostics(
        collectedDiagnostics.data() + pipelineAttemptDiagStart,
        collectedDiagnostics.size() - pipelineAttemptDiagStart);

    RetryPolicy *matchedPolicy = nullptr;
    std::optional<std::string> retryOption;
    for (const std::unique_ptr<RetryPolicy> &policy : policies) {
      if (std::optional<std::string> action =
              policy->onFailure(attemptDiagnostics, config)) {
        retryOption = std::move(action);
        matchedPolicy = policy.get();
        break;
      }
    }

    if (!retryOption || !matchedPolicy)
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "[BiShengHIR] Pipeline retry ("
                   << matchedPolicy->userVisibleRetryCause() << "): ";
      if (matchedPolicy->recordsCompileFallback())
        llvm::dbgs() << "disabling " << *retryOption;
      else
        llvm::dbgs() << "adjusting " << *retryOption;
      llvm::dbgs() << " and retrying\n";
    });
    if (matchedPolicy->recordsCompileFallback()) {
      if (llvm::none_of(appliedFallbacks, [&](const AppliedCompileFallback &a) {
            return a.disabledOption == *retryOption;
          }))
        appliedFallbacks.push_back(
            {*retryOption,
             std::string(matchedPolicy->userVisibleRetryCause())});
      emitFallbackNote(matchedPolicy->userVisibleRetryCause(), *retryOption);
    }

    collectedDiagnostics.resize(pipelineAttemptDiagStart);
  }
}
