//===- ConvertSCFToAffine.h - Pass entrypoint -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_
#define MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_

#include <memory>
#include <vector>

namespace mlir {
struct LogicalResult;
class MLIRContext;
class Pass;
class RewritePattern;

// Owning list of rewriting patterns.
class OwningRewritePatternList;
  
/// Collect a set of patterns to raise scf.for and scf.yield
/// to affine.for and affine.yield
void SCFForRaisingPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);
/// Creates a pass to convert scf.for and scf.yield ops
/// to affine.for and affine.yield
std::unique_ptr<Pass> createRaiseSCFForPass();
  
/// Collect a set of patterns to raise std.load and std.store
/// to affine.load and affine.store
void loadStoreRaisingPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);
/// Creates a pass to convert std.load and std.store to 
/// affine.load and affine.store.
std::unique_ptr<Pass> createRaiseLoadStorePass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_
