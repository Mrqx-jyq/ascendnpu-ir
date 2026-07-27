//===- EnhancedFusionDecision.cpp -- Smarter fusion decision pass --------===//
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
// This file implements the EnhancedFusionDecision pass, which analyzes
// data reuse and memory bandwidth to make smarter fusion decisions.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-enhanced-fusion"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

//===----------------------------------------------------------------------===//
// Data Reuse Analyzer
//===----------------------------------------------------------------------===//

/// Represents a producer-consumer relationship between ops.
struct ProducerConsumerEdge {
  Operation *producer;
  Operation *consumer;
  /// Number of elements reused from producer to consumer.
  int64_t reusedElements;
  /// Distance in number of ops between producer and consumer.
  int64_t distance;
  /// Whether the producer's output is fully consumed.
  bool isFullyConsumed;
};

/// Analyzes data reuse patterns within a function.
class DataReuseAnalyzer {
public:
  explicit DataReuseAnalyzer(func::FuncOp funcOp) : funcOp_(funcOp) {}

  /// Run analysis and return fusion recommendations.
  SmallVector<ProducerConsumerEdge> analyze() {
    SmallVector<ProducerConsumerEdge> edges;
    DenseMap<Value, Operation*> valueToProducer;

    // First pass: collect all value-to-producer mappings
    funcOp_.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        valueToProducer[result] = op;
      }
    });

    // Second pass: find consumer relationships
    funcOp_.walk([&](Operation *op) {
      llvm::SetVector<Operation*> producers;
      int64_t distance = 0;

      for (auto operand : op->getOperands()) {
        auto it = valueToProducer.find(operand);
        if (it != valueToProducer.end() && it->second != op) {
          producers.insert(it->second);
        }
      }

      // Compute reuse metrics for each producer
      for (auto *producer : producers) {
        ProducerConsumerEdge edge;
        edge.producer = producer;
        edge.consumer = op;

        // Estimate reused elements
        edge.reusedElements = 0;
        if (auto resultType = producer->getResult(0)
                .getType().dyn_cast<RankedTensorType>()) {
          int64_t numElements = 1;
          for (auto dim : resultType.getShape()) {
            if (dim > 0) numElements *= dim;
            else numElements = ShapedType::kDynamic;
          }
          if (numElements != ShapedType::kDynamic)
            edge.reusedElements = numElements;
        }

        // Estimate distance
        edge.distance = distance;
        edge.isFullyConsumed = true; // conservative

        edges.push_back(edge);
      }
    });

    return edges;
  }

  /// Score fusion opportunity (higher = better).
  double scoreFusionOpportunity(const ProducerConsumerEdge &edge,
                                double minReuseRatio) {
    if (edge.reusedElements <= 0)
      return 0.0;

    // Data reuse ratio: larger is better
    double reuseRatio = static_cast<double>(edge.reusedElements);

    // Distance penalty: closer is better
    double distancePenalty = 1.0 / (1.0 + edge.distance);

    return reuseRatio * distancePenalty;
  }

private:
  func::FuncOp funcOp_;
};

//===----------------------------------------------------------------------===//
// EnhancedFusionDecision Pass
//===----------------------------------------------------------------------===//

struct EnhancedFusionDecisionPass
    : public impl::EnhancedFusionDecisionBase<EnhancedFusionDecisionPass> {
  using EnhancedFusionDecisionBase::EnhancedFusionDecisionBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    LDBG("Analyzing fusion decisions for: " << funcOp.getSymName());

    if (!enableDataReuseAnalysis) {
      LDBG("Data reuse analysis disabled.");
      return;
    }

    // Step 1: Analyze data reuse patterns
    DataReuseAnalyzer analyzer(funcOp);
    auto edges = analyzer.analyze();

    LDBG("Found " << edges.size() << " producer-consumer edges.");

    // Step 2: Score fusion opportunities
    SmallVector<std::pair<double, ProducerConsumerEdge>> scoredEdges;
    for (auto &edge : edges) {
      double score = analyzer.scoreFusionOpportunity(edge, minDataReuseRatio);
      if (score > 0) {
        scoredEdges.push_back({score, edge});
      }
    }

    // Step 3: Annotate high-value fusion candidates
    // Sort by score descending
    std::sort(scoredEdges.begin(), scoredEdges.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    // Annotate top candidates with fusion hints
    int64_t fusionDepth = 0;
    for (auto &[score, edge] : scoredEdges) {
      if (fusionDepth >= maxFusionDepth)
        break;

      // Annotate producer with fusion hint
      edge.producer->setAttr("hfusion.fuse_with_consumer",
                             SymbolRefAttr::get(edge.consumer->getName()
                                 ? *edge.consumer->getName()
                                 : StringAttr::get(edge.consumer->getContext(),
                                                   "")));

      LDBG("Fusion candidate: " << edge.producer->getName() << " -> "
                                << edge.consumer->getName()
                                << " score=" << score);
      fusionDepth++;
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::hfusion::createEnhancedFusionDecisionPass(
    const EnhancedFusionDecisionOptions &options) {
  return std::make_unique<EnhancedFusionDecisionPass>(options);
}
