# GMC Educational Compiler

A small Python compiler project inspired by López, Karlsson, and Bientinesi, “Compilation of Generalized Matrix Chains with Symbolic Sizes,” CGO 2026.

Across its planned phases, the compiler will lex and parse a tiny matrix-chain DSL, infer matrix features, generate parenthesization variants, lower them to BLAS-like kernels, and emit Python functions with a FLOP-based runtime dispatcher.

## Phase 1: lexer only

Run the tests with:

```sh
python -m pytest -q
```

The implementation is staged across phases, from lexing and syntax analysis through runtime validation; kernels and code generation are original educational work, not copied from HPAC/gmc_code_gen.

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

Phase 7 prices lowered kernel steps with integer FLOP formulas over symbolic matrix dimensions and selects the cheapest essential variant for supplied sizes.

## Phase 8: Python code generation

Phase 8 emits ordinary Python variant and cost functions plus a runtime dispatcher. Phase 9 provides NumPy-backed runtime kernels and loads generated modules to validate numerical results.

## Phase 9: runtime execution

Phase 9 supplies NumPy-backed matrix kernels, deterministic test matrix construction, a left-to-right numerical reference, and residual checks against the generated cost-dispatched variant.

Run the full pipeline from the project root:

```sh
python -m gmc lex examples/simple.gmc
python -m gmc parse examples/simple.gmc
python -m gmc check examples/simple.gmc
python -m gmc infer examples/simple.gmc
python -m gmc variants examples/simple.gmc
python -m gmc lower examples/simple.gmc
python -m gmc cost examples/simple.gmc --sizes 10,8,8,5
python -m gmc compile examples/simple.gmc -o simple_gen.py
python -m gmc run examples/simple.gmc --sizes 10,8,8,5
```

### What this lab implemented

An original educational generalized matrix-chain compiler inspired by López, Karlsson, and Bientinesi, “Compilation of Generalized Matrix Chains with Symbolic Sizes,” CGO 2026.
Its phases cover lexing, parsing, semantic checks, feature inference, and parenthesization variants.
It lowers essential variants, estimates symbolic costs, generates Python dispatch modules, and validates numerical results.
The final runtime uses NumPy-backed kernels and compares the selected result with a direct reference product.
The compiler structure and code are original educational work, not a copy of HPAC/gmc_code_gen.