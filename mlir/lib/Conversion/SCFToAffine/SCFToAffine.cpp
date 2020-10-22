//===- SCFToStandard.cpp - ControlFlow to CFG conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.for, scf.if and loop.terminator
// ops into standard CFG ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h" // JC: TO DELETE

#include "mlir/Conversion/SCFToAffine/SCFToAffine.h"
#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class SCFToAffineTarget : public ConversionTarget {
public:
  explicit SCFToAffineTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  bool isDynamicallyLegal(Operation *op) const override {
    // JC: todo affine load/store in irregular loops

    if (auto forOp = dyn_cast<scf::ForOp>(op)){
      auto step = forOp.step();
      auto lb = forOp.lowerBound();
      auto ub = forOp.upperBound();
      return !(SSACheck(&step) && SSACheck(&lb) && SSACheck(&ub));
    }
    else
      return true;
  }

  bool SSACheck(Value *value) const {
    if (isa<ConstantOp>(value->getDefiningOp()))
      // 4. the result of a constant operation ,
      return true;
    else if (auto indexCastOp = dyn_cast<IndexCastOp>(value->getDefiningOp()))
      // 4.5 the result of a constant operation casted from another type
      if (isa<ConstantOp>(indexCastOp.getOperand().getDefiningOp()))
        return true;
    return false;
  }

};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct SCFToAffinePass : public SCFToAffineBase<SCFToAffinePass> {
  void runOnOperation() override;
};

struct LoadRaising : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};
struct StoreRaising : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override;
};


struct SCFForRaising : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

// JC to do - raise it to affine.for if possible
LogicalResult SCFForRaising::matchAndRewrite(ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();
  
  /*auto step = forOp.step();
  auto lb = forOp.lowerBound();
  auto ub = forOp.upperBound();
  auto indVar = forOp.getInductionVar();*/




  /*OperationState &result,
  ValueRange lbOperands, AffineMap lbMap,
  ValueRange ubOperands, AffineMap ubMap, int64_t step,
  ValueRange iterArgs, BodyBuilderFn bodyBuilder

  auto f = rewriter.create<AffineForOp>(loc, result, lbOperands, lbMap, ubOperands,ubMap, iterArgs, bodyBuilder);

  rewriter.eraseOp(forOp);*/


  /*
    Location loc = op.getLoc();
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step = rewriter.create<ConstantIndexOp>(loc, op.getStep());
    auto f = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    rewriter.eraseBlock(f.getBody());
    rewriter.inlineRegionBefore(op.region(), f.region(), f.region().end());
    rewriter.eraseOp(op);
    return success();
  */

  // Start by splitting the block containing the 'scf.for' into two parts.
  // The part before will get the init code, the part after will be the end
  // point.
  auto *initBlock = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

  // Use the first block of the loop body as the condition block since it is the
  // block that has the induction variable and loop-carried values as arguments.
  // Split out all operations from the first block into a new block. Move all
  // body blocks from the loop body region to the region containing the loop.
  auto *conditionBlock = &forOp.region().front();
  auto *firstBodyBlock =
      rewriter.splitBlock(conditionBlock, conditionBlock->begin());
  auto *lastBodyBlock = &forOp.region().back();
  rewriter.inlineRegionBefore(forOp.region(), endBlock);
  auto iv = conditionBlock->getArgument(0);

  // Append the induction variable stepping logic to the last body block and
  // branch back to the condition block. Loop-carried values are taken from
  // operands of the loop terminator.
  Operation *terminator = lastBodyBlock->getTerminator();
  rewriter.setInsertionPointToEnd(lastBodyBlock);
  auto step = forOp.step();
  auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
  if (!stepped)
    return failure();

  SmallVector<Value, 8> loopCarried;
  loopCarried.push_back(stepped);
  loopCarried.append(terminator->operand_begin(), terminator->operand_end());
  rewriter.create<BranchOp>(loc, conditionBlock, loopCarried);
  rewriter.eraseOp(terminator);

  // Compute loop bounds before branching to the condition.
  rewriter.setInsertionPointToEnd(initBlock);
  Value lowerBound = forOp.lowerBound();
  Value upperBound = forOp.upperBound();
  if (!lowerBound || !upperBound)
    return failure();

  // The initial values of loop-carried values is obtained from the operands
  // of the loop operation.
  SmallVector<Value, 8> destOperands;
  destOperands.push_back(lowerBound);
  auto iterOperands = forOp.getIterOperands();
  destOperands.append(iterOperands.begin(), iterOperands.end());
  rewriter.create<BranchOp>(loc, conditionBlock, destOperands);

  // With the body block done, we can fill in the condition block.
  rewriter.setInsertionPointToEnd(conditionBlock);
  auto comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);

  rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                ArrayRef<Value>(), endBlock, ArrayRef<Value>());
  // The result of the loop operation is the values of the condition block
  // arguments except the induction variable on the last iteration.
  rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());
  return success();
}

LogicalResult LoadRaising::matchAndRewrite(LoadOp loadOp,
                                           PatternRewriter &rewriter) const {
  Location loc = loadOp.getLoc();
  
  // rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, op.getMemRef(), *resultOperands);
  // rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, op.getMemRef(), *resultOperands);
  return success();
}

void mlir::analyzeAndTransformMemoryOps(
  OwningRewritePatternList &patterns, MLIRContext *ctx) {
    patterns.insert<SCFForRaising, LoadRaising>(ctx);
  }

void SCFToAffinePass::runOnOperation() {
  OwningRewritePatternList patterns;
  analyzeAndTransformMemoryOps(patterns, &getContext());
  // Configure conversion to lower out scf.for, scf.if and scf.parallel.
  // Anything else is fine.
  SCFToAffineTarget target(getContext());
  target.addLegalDialect<SCFDialect, AffineDialect>();

  target.addDynamicallyLegalOp<scf::ForOp>();
  
  target.addDynamicallyLegalOp<LoadOp/*, StoreOp*/>();

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createRaiseToAffinePass() {
  return std::make_unique<SCFToAffinePass>();
}
