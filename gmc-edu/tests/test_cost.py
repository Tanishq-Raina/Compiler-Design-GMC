from pathlib import Path
import subprocess
import sys

import pytest

from gmc.cost import (
    CostError,
    dump_priced,
    flops,
    flops_symbolic,
    parse_sizes,
    pick_best,
    price_source,
)
from gmc.features import Feature
from gmc.lower import OpState


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _state(structure: str, property: str = "INV", inverted: bool = False) -> OpState:
    return OpState(structure, Feature(structure, property), inverted, structure)


def _price_example(filename: str):
    source = (PROJECT_ROOT / "examples" / filename).read_text(encoding="utf-8")
    return price_source(source)


def test_gemm_numeric_and_symbolic_cost() -> None:
    assert flops("GEMM", 2, 6, 3, _state("G"), _state("G")) == 72
    assert flops_symbolic("GEMM", 0, 1, 2, _state("G"), _state("G")) == (
        "2*q0*q1*q2"
    )


@pytest.mark.parametrize(
    ("kernel", "left", "right", "dimensions", "expected"),
    [
        ("SYMM", _state("S"), _state("G"), (2, 6, 3), 48),
        ("SYMM", _state("G"), _state("S"), (2, 6, 3), 144),
        ("TRMM", _state("L"), _state("G"), (2, 6, 3), 24),
        ("TRMM", _state("G"), _state("U"), (2, 6, 3), 72),
        ("SYSYMM", _state("S"), _state("S"), (2, 6, 3), 16),
        ("TRSYMM", _state("L"), _state("S"), (2, 6, 3), 8),
        ("TRTRMM", _state("L"), _state("L"), (2, 6, 3), 2),
        ("TRTRMM", _state("L"), _state("U"), (2, 6, 3), 5),
    ],
)
def test_multiply_kernel_flop_table(
    kernel: str,
    left: OpState,
    right: OpState,
    dimensions: tuple[int, int, int],
    expected: int,
) -> None:
    assert flops(kernel, *dimensions, left, right) == expected


@pytest.mark.parametrize(
    ("kernel", "left", "right", "expected"),
    [
        ("TRSM", _state("L", inverted=True), _state("G"), 24),
        ("TRSM", _state("G"), _state("L", inverted=True), 72),
        ("TRSYSV", _state("L", inverted=True), _state("S"), 8),
        ("TRTRSV", _state("L", inverted=True), _state("L"), 2),
        ("TRTRSV", _state("L", inverted=True), _state("U"), 8),
        ("GEGESV", _state("G", inverted=True), _state("G"), 53),
        ("GEGESV", _state("G"), _state("G", inverted=True), 288),
        ("SYGESV", _state("S", inverted=True), _state("G"), 50),
        ("POGESV", _state("S", "SPD", True), _state("G"), 50),
        ("GESYSV", _state("G", inverted=True), _state("S"), 21),
        ("SYSYSV", _state("S", inverted=True), _state("S"), 18),
        ("POSYSV", _state("S", "SPD", True), _state("S"), 18),
        ("SYTRSV", _state("S", inverted=True), _state("L"), 18),
        ("GETRSV", _state("G", inverted=True), _state("L"), 16),
        ("POTRSV", _state("S", "SPD", True), _state("L"), 13),
        ("POTRSV", _state("S", "SPD", True), _state("U"), 18),
    ],
)
def test_solve_kernel_flop_table(
    kernel: str, left: OpState, right: OpState, expected: int
) -> None:
    assert flops(kernel, 2, 6, 3, left, right) == expected


def test_unknown_kernel_and_invalid_sizes_raise_cost_error() -> None:
    with pytest.raises(CostError):
        flops("UNKNOWN", 2, 3, 4, _state("G"), _state("G"))
    with pytest.raises(CostError, match="expected 4 sizes, got 3"):
        parse_sizes("10,8,8", 3)
    with pytest.raises(CostError, match="expected 4 sizes, got 0"):
        parse_sizes("", 3)
    with pytest.raises(CostError, match="positive"):
        parse_sizes("10,0,8,5", 3)


def test_four_factor_worked_costs_and_best_variant() -> None:
    priced = _price_example("four.gmc")
    sizes = [2, 3, 4, 5, 6]
    by_label = {variant.lowered.h_label: variant for variant in priced}
    assert by_label["E0"].eval(sizes) == 248
    assert by_label["E2"].eval(sizes) == 384
    assert by_label["E4"].eval(sizes) == 456
    assert priced[pick_best(priced, sizes)].lowered.h_label == "E0"


def test_simple_worked_costs_and_best_variant() -> None:
    priced = _price_example("simple.gmc")
    sizes = [10, 8, 8, 5]
    by_label = {variant.lowered.h_label: variant for variant in priced}
    assert by_label["E0"].eval(sizes) == 1440
    assert by_label["E1"].eval(sizes) == 1120
    assert priced[pick_best(priced, sizes)].lowered.h_label == "E1"


def test_good_spd_worked_cost() -> None:
    priced = _price_example("good_spd.gmc")
    assert priced[0].eval([4, 4, 7]) == 245
    assert priced[0].priced_steps[0].symbolic == "(q0*q0*q0)//3+2*q0*q0*q2"


def test_triangular_worked_cost_uses_integer_division() -> None:
    priced = _price_example("infer_tri.gmc")
    variant = next(item for item in priced if item.lowered.h_label == "E0")
    assert [step.step.kernel for step in variant.priced_steps] == [
        "TRTRMM",
        "TRTRMM",
    ]
    assert variant.eval([5, 5, 5, 5]) == 124


def test_symbolic_and_numeric_dumps() -> None:
    priced = _price_example("simple.gmc")
    symbolic = dump_priced(priced, None)
    assert "cost =" in symbolic
    assert "2*" in symbolic or "q0" in symbolic

    numeric = dump_priced(priced, [10, 8, 8, 5])
    assert "flops =" in numeric
    assert "best =" in numeric


def test_variant_evaluation_checks_size_vector_length() -> None:
    variant = _price_example("good_spd.gmc")[0]
    with pytest.raises(CostError, match="expected 3 sizes, got 2"):
        variant.eval([4, 4])


def test_cost_cli_with_sizes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gmc",
            "cost",
            "examples/four.gmc",
            "--sizes",
            "2,3,4,5,6",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "248" in result.stdout
    assert "best = E0" in result.stdout