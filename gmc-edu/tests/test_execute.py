from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gmc.codegen import generate_source
from gmc.execute import (
    ExecuteError,
    build_mats,
    load_generated,
    run_source,
)
from gmc.parser import parse
from gmc.semantics import check
from gmc import runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source(filename: str) -> str:
    return (PROJECT_ROOT / "examples" / filename).read_text(encoding="utf-8")


def test_runtime_multiply_transpose_and_solves() -> None:
    rng = np.random.default_rng(11)
    left = rng.standard_normal((2, 3))
    right = rng.standard_normal((3, 4))
    np.testing.assert_allclose(runtime.GEMM(left, right), left @ right)
    np.testing.assert_allclose(runtime.transpose(left), left.T)

    coefficient = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    rhs = rng.standard_normal((4, 3))
    np.testing.assert_allclose(
        runtime.TRSM(coefficient, rhs, inverted_left=True),
        np.linalg.solve(coefficient, rhs),
    )

    lhs = rng.standard_normal((3, 4))
    right_coefficient = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    np.testing.assert_allclose(
        runtime.TRSM(lhs, right_coefficient, inverted_left=False),
        lhs @ np.linalg.inv(right_coefficient),
    )


def test_four_factor_end_to_end_run() -> None:
    result = run_source(_source("four.gmc"), [2, 3, 4, 5, 6], seed=0)
    assert result.label == "E0"
    assert result.flops == 248
    assert result.residual < 1e-10
    assert result.value.shape == (2, 6)


def test_simple_run_picks_e1_and_matches_dispatch() -> None:
    source = _source("simple.gmc")
    sizes = [10, 8, 8, 5]
    result = run_source(source, sizes, seed=0)
    assert result.label == "E1"
    assert result.flops == 1120
    assert result.residual < 1e-8
    assert result.value.shape == (10, 5)

    program = parse(source)
    checked = check(program)
    rng = np.random.default_rng(0)
    mats = build_mats(program, checked.symbols, sizes, rng)
    namespace = load_generated(generate_source(source))
    dispatched = namespace["dispatch"](mats, sizes)
    variant_e1 = namespace["VARIANTS"][namespace["LABELS"].index("E1")](mats)
    np.testing.assert_allclose(dispatched, variant_e1)


def test_good_spd_run() -> None:
    result = run_source(_source("good_spd.gmc"), [4, 4, 7], seed=0)
    assert result.label == "E0"
    assert result.flops == 245
    assert result.residual < 1e-8
    assert result.value.shape == (4, 7)


def test_triangular_run() -> None:
    result = run_source(_source("infer_tri.gmc"), [5, 5, 5, 5], seed=0)
    assert result.residual < 1e-8
    assert result.value.shape == (5, 5)


def test_run_cli_outputs_result() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gmc",
            "run",
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
    assert "picked E0" in result.stdout
    assert "residual=" in result.stdout


def test_inverse_use_requires_square_size_instance() -> None:
    with pytest.raises(ExecuteError, match="inv.*square"):
        run_source(_source("simple.gmc"), [10, 8, 7, 5], seed=0)
