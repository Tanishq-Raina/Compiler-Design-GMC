# GMC Educational Compiler

A small Python compiler project inspired by López, Karlsson, and Bientinesi, “Compilation of Generalized Matrix Chains with Symbolic Sizes,” CGO 2026.

Across its planned phases, the compiler will lex and parse a tiny matrix-chain DSL, infer matrix features, generate parenthesization variants, lower them to BLAS-like kernels, and emit Python functions with a FLOP-based runtime dispatcher.

## Phase 1: lexer only

Run the tests with:

```sh
python -m pytest -q
```

Phase 1 implemented the lexer; later phases add syntax analysis, semantic checks, feature inference, variants, lowering, symbolic costs, and original code generation. Runtime kernels come later and will not be copied from HPAC/gmc_code_gen.

## Phase 2: syntax analysis

Phase 2 adds a recursive-descent parser and a small AST for declarations and flat product factors; it performs syntax checks only, with no symbol-table or matrix-feature analysis.

## Phase 3: semantic analysis

Phase 3 validates declaration combinations, duplicate names, declared variable uses, and inverse eligibility through a separate symbol table; it does not mutate the AST or infer product features.

## Phase 4: feature inference

Phase 4 maps declared matrix properties to canonical feature classes and infers result features with the structure and property tables; its left-to-right chain result is only a preview, not a chosen parenthesization.

## Phase 5: parenthesization variants

Phase 5 generates all full parenthesizations plus the fanning-out and size-equivalence essential subsets; it reports candidate trees without selecting a best association or lowering to kernels.

## Phase 6: kernel lowering

Phase 6 lowers each essential association to a sequence of BLAS-like kernel labels with symbolic intermediate states; it does not calculate costs or emit runnable code.

## Phase 7: symbolic cost model

Phase 7 prices lowered kernel steps with integer FLOP formulas over symbolic matrix dimensions and can select the cheapest essential variant for supplied sizes; code generation remains a later phase.

## Phase 8: Python code generation

Phase 8 emits ordinary Python variant and cost functions plus a runtime dispatcher; kernel implementations remain Phase 9 stubs, and generated modules are not executed by the compiler.