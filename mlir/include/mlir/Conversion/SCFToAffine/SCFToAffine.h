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

/// Collect a set of patterns to lower from scf.for, scf.if, and
/// loop.terminator to CFG operations within the Standard dialect, in particular
/// convert structured control flow into CFG branch-based control flow.
void analyzeAndTransformMemoryOps(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);

/// Creates a pass to convert std.load and std.load ops to affine.load and affine.store ops.
std::unique_ptr<Pass> createRaiseToAffinePass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOAFFINE_SCFTOAFFINE_H_
