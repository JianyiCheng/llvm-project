//===- SCFToAffine.cpp - Raising SCF to Affine conversion ------------------===//
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


using namespace mlir;
using namespace mlir::scf;


namespace {

// Conversion Target
class SCFToAffineTarget : public ConversionTarget {
public:
  explicit SCFToAffineTarget(MLIRContext &context)
      : ConversionTarget(context) {}

  bool isDynamicallyLegal(Operation *op) const override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)){
      if (forOp.getNumResults() > 0) // JC: affine.for with results seems not supported?
        return true;
      else if (auto cst = forOp.step().getDefiningOp<ConstantIndexOp>()){
        // step > 0 already verified in SCF dialect
        auto lb = forOp.lowerBound();
        auto ub = forOp.upperBound();
        // if (!(SSACheck(lb, forOp) && SSACheck(ub, forOp)))
        if (!(isValidSymbol(lb) && isValidSymbol(ub)))
          return true;
        else {
          return false;
        }
      }
      else    // only constant step supported in affine.for
        return true;
    }
    else if (auto loadOp = dyn_cast<LoadOp>(op)){
      /*while (auto op = loadOp.getParentOp()){
        if (isa<AffineForOp>(op)){
          for (auto i = 0; i < loadOp.getNumOperands)
        }
      }*/
      return false;
      // return isValidDim(loadOp.getOperand(1));
    }
    else
      return true;
  }

// Rewriting Pass
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

// Raise scf.for to affine.for & replace scf.yield with affine.yield
LogicalResult SCFForRaising::matchAndRewrite(ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();
  auto ctx = rewriter.getContext();
  auto step = forOp.step();
  auto lb = forOp.lowerBound();
  auto ub = forOp.upperBound();
  auto indVar = forOp.getInductionVar();
  auto results = forOp.getResults();
  auto iterOperands = forOp.getIterOperands();
  auto iterArgs = forOp.getRegionIterArgs();
  AffineForOp::BodyBuilderFn bodyBuilder;
  int stepNum = dyn_cast<ConstantOp>(step.getDefiningOp()).getValue().cast<IntegerAttr>().getInt();
  AffineMap directSymbolMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, rewriter.getContext()));
  auto f = rewriter.create<AffineForOp>(loc, lb, directSymbolMap, ub, directSymbolMap, stepNum, iterArgs, bodyBuilder);
  rewriter.eraseBlock(f.getBody());
  Operation *loopTerminator = forOp.region().back().getTerminator();
  ValueRange terminatorOperands = loopTerminator->getOperands();
  rewriter.setInsertionPointToEnd(&forOp.region().back());
  rewriter.create<AffineYieldOp>(loc, terminatorOperands);
  
  rewriter.inlineRegionBefore(forOp.region(), f.region(), f.region().end());
  rewriter.eraseOp(loopTerminator);
  rewriter.eraseOp(forOp);

  return success();
}

// Extract the affine expression from a number of std instructions
AffineExpr getAffineExpr(Value value, AffineExpr expr, bool *collect,
                                           PatternRewriter &rewriter){
  if (auto constant = dyn_cast_or_null<ConstantOp>(value.getDefiningOp()))
    return getAffineConstantExpr(constant.getValue().cast<IntegerAttr>().getInt(), rewriter.getContext());
  else if (isValidSymbol(value) && !isValidDim(value))
    return getAffineSymbolExpr(0, rewriter.getContext());
  else
    return getAffineDimExpr(0, rewriter.getContext());
}

// Raise std.load to affine.load
LogicalResult LoadRaising::matchAndRewrite(LoadOp loadOp,
                                           PatternRewriter &rewriter) const {
  Location loc = loadOp.getLoc();
  AffineExpr expr;
  SmallVector<Value, 1> lhs_indices{};
  for (int i = 1; i < loadOp.getNumOperands(); i++){
    bool collect = false;
    AffineExpr exprInit;
    lhs_indices = {loadOp.getOperand(i)}; // 1D array for
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
  // Configure conversion to raise scf.for, std.load and std.store.
  // Anything else is fine.
  SCFToAffineTarget target(getContext());
  target.addLegalDialect<SCFDialect, AffineDialect>();
  target.addDynamicallyLegalOp<scf::ForOp>();
  target.addDynamicallyLegalOp<LoadOp/*, StoreOp*/>();  // JC TODO: STORE

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createRaiseToAffinePass() {
  return std::make_unique<SCFToAffinePass>();
}
