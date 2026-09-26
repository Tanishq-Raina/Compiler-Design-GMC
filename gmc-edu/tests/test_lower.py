from pathlib import Path
import subprocess
import sys

import pytest

from gmc.features import Feature
from gmc.lower import (
    LowerError,
    OpState,
    assign_kernel,
    both_inverted,
    dump_lowered,
    lower_file,
    rewrite_association,
    should_propagate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _state(
    structure: str,
    property: str = "INV",
    inverted: bool = False,
    source: str = "M",
) -> OpState:
    return OpState(source, Feature(structure, property), inverted, source)


@pytest.mark.parametrize(
    ("left", "right", "kernel"),
    [
        ("G", "G", "GEMM"),
        ("S", "G", "SYMM"),
        ("L", "G", "TRMM"),
        ("U", "G", "TRMM"),
        ("S", "S", "SYSYMM"),
        ("L", "S", "TRSYMM"),
        ("L", "L", "TRTRMM"),
        ("U", "L", "TRTRMM"),
        ("D", "G", "TRMM"),
    ],
)
def test_assign_kernel_no_inversions(left: str, right: str, kernel: str) -> None:
    assert assign_kernel(_state(left), _state(right)) == kernel


@pytest.mark.parametrize(
    ("inverted", "other", "property", "kernel"),
    [
        ("L", "G", "INV", "TRSM"),
        ("S", "G", "SPD", "POGESV"),
        ("S", "G", "INV", "SYGESV"),
        ("G", "G", "INV", "GEGESV"),
        ("L", "L", "INV", "TRTRSV"),
    ],
)
def test_assign_kernel_one_inverted_operand(
    inverted: str, other: str, property: str, kernel: str
) -> None:
    assert assign_kernel(
        _state(inverted, property, True), _state(other)
    ) == kernel


def test_assign_kernel_rejects_two_inverted_operands() -> None:
    with pytest.raises(LowerError):
        assign_kernel(_state("G", inverted=True), _state("G", inverted=True))


def test_rewrite_two_inverted_operands_swaps_and_clears() -> None:
    left = _state("G", inverted=True, source="inv(A)")
    right = _state("G", inverted=True, source="inv(B)")
    assert both_inverted(left, right)
    rewritten_left, rewritten_right, result_inverted = rewrite_association(left, right)
    assert (rewritten_left.source, rewritten_right.source) == ("inv(B)", "inv(A)")
    assert not rewritten_left.inverted
    assert not rewritten_right.inverted
    assert result_inverted
    assert assign_kernel(rewritten_left, rewritten_right) == "GEMM"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (_state("L"), _state("G", inverted=True), True),
        (_state("G", inverted=True), _state("L"), True),
        (_state("L", inverted=True), _state("G"), False),
        (_state("G"), _state("G", inverted=True), False),
    ],
)
def test_should_propagate(left: OpState, right: OpState, expected: bool) -> None:
    assert should_propagate(left, right) is expected


def test_propagation_moves_inverse_to_triangular_operand() -> None:
    left, right, result_inverted = rewrite_association(
        _state("L", source="L"),
        _state("G", inverted=True, source="inv(G)"),
    )
    assert (left.feature.structure, left.inverted) == ("G", False)
    assert (right.feature.structure, right.inverted) == ("L", True)
    assert result_inverted
    assert assign_kernel(left, right) == "TRSM"


def test_lowering_propagation_keeps_result_inverted_flag() -> None:
    from gmc.lower import lower_source

    plan = lower_source(
        "Matrix L <Lower, FullRank>; Matrix G <Dense, FullRank>; L * inv(G)"
    )
    step = plan.variants[0].steps[0]
    assert step.kernel == "TRSM"
    assert (step.left.feature.structure, step.left.inverted) == ("G", False)
    assert (step.right.feature.structure, step.right.inverted) == ("L", True)
    assert step.result_inverted
    assert plan.variants[0].result.inverted


def test_lowering_two_inverses_rewrites_to_multiply_and_inverted_result() -> None:
    from gmc.lower import lower_source

    plan = lower_source(
        "Matrix A <Dense, FullRank>; Matrix B <Dense, FullRank>; inv(A) * inv(B)"
    )
    step = plan.variants[0].steps[0]
    assert step.kernel == "GEMM"
    assert (step.left.source, step.right.source) == ("inv(B)", "inv(A)")
    assert step.result_inverted


def test_simple_file_lowers_both_essential_variants() -> None:
    plan = lower_file(PROJECT_ROOT / "examples" / "simple.gmc")
    assert [variant.h_label for variant in plan.variants] == ["E0", "E1"]
    assert [[step.kernel for step in variant.steps] for variant in plan.variants] == [
        ["TRSM", "GEMM"],
        ["TRSM", "GEMM"],
    ]
    output = dump_lowered(plan)
    assert "T1 = TRSM" in output
    assert "T2 = GEMM" in output


def test_four_factor_lowering_is_post_order() -> None:
    plan = lower_file(PROJECT_ROOT / "examples" / "four.gmc")
    assert len(plan.variants) == 5
    assert all(
        step.kernel == "GEMM"
        for variant in plan.variants
        for step in variant.steps
    )
    variant_e0 = next(variant for variant in plan.variants if variant.h_label == "E0")
    variant_e2 = next(variant for variant in plan.variants if variant.h_label == "E2")
    assert len(variant_e0.steps) == 3
    assert [(step.left.source, step.right.source) for step in variant_e2.steps] == [
        ("A", "B"),
        ("C", "D"),
        ("T1", "T2"),
    ]


def test_spd_inverse_uses_positive_definite_solver() -> None:
    plan = lower_file(PROJECT_ROOT / "examples" / "good_spd.gmc")
    assert len(plan.variants) == 1
    assert [step.kernel for step in plan.variants[0].steps] == ["POGESV"]


def test_triangular_example_uses_triangular_multiply_kernels() -> None:
    plan = lower_file(PROJECT_ROOT / "examples" / "infer_tri.gmc")
    variant_e0 = next(variant for variant in plan.variants if variant.h_label == "E0")
    assert [step.kernel for step in variant_e0.steps] == ["TRTRMM", "TRTRMM"]


def test_lowered_dump_is_deterministic() -> None:
    path = PROJECT_ROOT / "examples" / "simple.gmc"
    assert dump_lowered(lower_file(path)) == dump_lowered(lower_file(path))


def test_lower_cli_prints_both_simple_variants() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gmc", "lower", "examples/simple.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "variant E0" in result.stdout
    assert "variant E1" in result.stdout