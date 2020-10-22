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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
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
// jc to delete
#include <sstream>

using namespace mlir;
using namespace mlir::scf;



bool isConstant(Value value){
  if (!value.getDefiningOp())
    return false;
  else if (isa<ConstantOp>(value.getDefiningOp()))
      return true;
  else if (auto indexCastOp = dyn_cast<IndexCastOp>(value.getDefiningOp())){
      // the result of a constant operation casted from another type
    if (isa<ConstantOp>(indexCastOp.getOperand().getDefiningOp()))
        return true;
  }
  return false;
}

int getConstantInt(Value value){
  assert(isConstant(value));
  if (auto constant = dyn_cast<ConstantOp>(value.getDefiningOp()))
      return constant.getValue().cast<IntegerAttr>().getInt();
  else if (auto indexCastOp = dyn_cast<IndexCastOp>(value.getDefiningOp())){
      // the result of a constant operation casted from another type
    if (auto castConstant = dyn_cast<ConstantOp>(indexCastOp.getOperand().getDefiningOp()))
        return castConstant.getValue().cast<IntegerAttr>().getInt();
  }
  return NULL;
}

namespace {

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class SCFToAffineTarget : public ConversionTarget {
public:
  explicit SCFToAffineTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  bool isDynamicallyLegal(Operation *op) const override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)){
      auto step = forOp.step();
      auto lb = forOp.lowerBound();
      auto ub = forOp.upperBound();
      return !(SSACheck(step) && SSACheck(lb) && SSACheck(ub));
    }
    else if (auto loadOp = dyn_cast<LoadOp>(op)){
      return false;
    }
    else
      return true;
  }

  bool SSACheck(Value value) const {
    if (isConstant(value))  // 4. the result of a constant operation ,
      return true;
    else
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
  
  auto step = forOp.step();
  auto lb = forOp.lowerBound();
  auto ub = forOp.upperBound();
  auto indVar = forOp.getInductionVar();
  auto results = forOp.getResults();
  auto iterOperands = forOp.getIterOperands();
  auto iterArgs = forOp.getRegionIterArgs();
  AffineForOp::BodyBuilderFn bodyBuilder;
  // if constant bound
  // int stepNum = (isa<ConstantOp>(step.getDefiningOp())) ? step.getDefiningOp().getValue():dyn_cast<IndexCastOp>(step.getDefiningOp()).getOperand().getValue();
  // int lbNum = (isa<ConstantOp>(lb.getDefiningOp())) ? lb.getDefiningOp().getValue():dyn_cast<IndexCastOp>(lb.getDefiningOp()).getOperand().getValue();
  // int ubNum = (isa<ConstantOp>(ub.getDefiningOp())) ? ub.getDefiningOp().getValue():dyn_cast<IndexCastOp>(ub.getDefiningOp()).getOperand().getValue();
  auto f = rewriter.create<AffineForOp>(loc, 0, 1000, 1, iterArgs, bodyBuilder);
  rewriter.eraseBlock(f.getBody());
  Operation *loopTerminator = forOp.region().back().getTerminator();
  ValueRange terminatorOperands = loopTerminator->getOperands();
  rewriter.setInsertionPointToEnd(&forOp.region().back());
  rewriter.create<AffineYieldOp>(loc, terminatorOperands);
  rewriter.eraseOp(loopTerminator);
  rewriter.inlineRegionBefore(forOp.region(), f.region(), f.region().end());
  rewriter.eraseOp(forOp);

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
  /*auto *initBlock = rewriter.getInsertionBlock();
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
  rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());*/
  return success();
}

AffineExpr getAffineExpr(Value v, AffineExpr expr, bool *collect,
                                           PatternRewriter &rewriter){
  if (isConstant(v))
    return getAffineConstantExpr(getConstantInt(v), rewriter.getContext());
  else if (isValidSymbol(v) && !isValidDim(v))
    return getAffineSymbolExpr(0, rewriter.getContext());
  else
    return getAffineDimExpr(0, rewriter.getContext());
}

LogicalResult LoadRaising::matchAndRewrite(LoadOp loadOp,
                                           PatternRewriter &rewriter) const {
  Location loc = loadOp.getLoc();
  AffineExpr expr;
  SmallVector<Value, 1> lhs_indices{};
  for (int i = 1; i < loadOp.getNumOperands(); i++){
    bool collect = false;
    AffineExpr exprInit;
    // std::string resultName;
    // llvm::raw_string_ostream string_stream(resultName);
    // loadOp.getOperand(i).print(string_stream);
    // return loadOp.emitOpError("debug found: " + resultName + "\n");
    lhs_indices = {loadOp.getOperand(i)};
    /*AffineExpr*/ expr = getAffineExpr(loadOp.getOperand(i), exprInit, &collect, rewriter);
  }
   
  AffineMap loadMap = AffineMap::get(1, 0, expr);
  rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, loadOp.getMemRef(), loadMap, lhs_indices);
  // rewriter.eraseOp(loadOp);
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
