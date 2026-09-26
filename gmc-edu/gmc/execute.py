"""Build concrete matrices and run generated GMC programs for validation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from gmc.ast import Operand, Program, UnaryOp, Var
from gmc.codegen import generate_source
from gmc.cost import CostError, price_source
from gmc.parser import parse
from gmc.semantics import CheckedProgram, Symbol, base_var, check_source


class ExecuteError(Exception):
    """Invalid matrix shapes or execution setup."""


@dataclass
class RunResult:
    label: str
    value: np.ndarray
    reference: np.ndarray
    residual: float
    flops: int
    q: list[int]


def factor_shapes(n: int, q: list[int]) -> list[tuple[int, int]]:
    if n < 1:
        raise ExecuteError("a matrix chain must contain at least one factor")
    if len(q) != n + 1:
        raise CostError(f"expected {n + 1} sizes, got {len(q)}")
    if any(type(size) is not int or size <= 0 for size in q):
        raise CostError("sizes must be positive integers")
    return [(q[index], q[index + 1]) for index in range(n)]


def make_matrix(
    structure: str,
    property: str,
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = rng.standard_normal((rows, cols))

    if property == "SPD":
        _require_square(structure, property, rows, cols)
        matrix = rng.standard_normal((rows, rows))
        matrix = matrix.T @ matrix + rows * np.eye(rows)
    elif property == "Orthogonal":
        _require_square(structure, property, rows, cols)
        matrix, _ = np.linalg.qr(matrix)
    elif structure == "Lower":
        matrix = np.tril(matrix)
    elif structure == "UnitLower":
        _require_square(structure, property, rows, cols)
        matrix = np.tril(matrix, -1) + np.eye(rows)
    elif structure == "Upper":
        matrix = np.triu(matrix)
    elif structure == "UnitUpper":
        _require_square(structure, property, rows, cols)
        matrix = np.triu(matrix, 1) + np.eye(rows)
    elif structure == "Diagonal":
        diagonal_size = min(rows, cols)
        diagonal = np.zeros((rows, cols), dtype=np.float64)
        indices = np.arange(diagonal_size)
        diagonal[indices, indices] = rng.standard_normal(diagonal_size)
        matrix = diagonal
    elif structure == "Symmetric":
        _require_square(structure, property, rows, cols)
        matrix = (matrix + matrix.T) / 2
    elif structure != "Dense":
        raise ExecuteError(f"unknown matrix structure {structure!r}")

    if property == "FullRank" and structure not in {"UnitLower", "UnitUpper"}:
        if rows == cols:
            matrix = matrix + 0.5 * np.eye(rows)
    elif property not in {"None", "FullRank", "SPD", "Orthogonal"}:
        raise ExecuteError(f"unknown matrix property {property!r}")

    return np.asarray(matrix, dtype=np.float64)


def _require_square(structure: str, property: str, rows: int, cols: int) -> None:
    if rows != cols:
        raise ExecuteError(
            f"{structure}/{property} matrix must be square, got {rows}x{cols}"
        )


def build_mats(
    program: Program,
    symbols: dict[str, Symbol],
    q: list[int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    shapes = factor_shapes(len(program.factors), q)
    matrices: dict[str, np.ndarray] = {}
    declared_shapes: dict[str, tuple[int, int]] = {}

    for index, factor in enumerate(program.factors):
        variable = base_var(factor)
        transpose_parity = _transpose_parity(factor)
        factor_rows, factor_cols = shapes[index]
        base_shape = (
            (factor_cols, factor_rows)
            if transpose_parity
            else (factor_rows, factor_cols)
        )
        if _contains_inverse(factor) and base_shape[0] != base_shape[1]:
            raise ExecuteError(
                f"inverse use of {variable.name} requires a square matrix; "
                f"got {base_shape[0]}x{base_shape[1]}"
            )

        previous_shape = declared_shapes.get(variable.name)
        if previous_shape is not None and previous_shape != base_shape:
            raise ExecuteError(
                f"matrix {variable.name} is used with incompatible shapes "
                f"{previous_shape} and {base_shape}"
            )
        if previous_shape is not None:
            continue

        try:
            symbol = symbols[variable.name]
        except KeyError as error:
            raise ExecuteError(f"missing symbol {variable.name!r}") from error
        declared_shapes[variable.name] = base_shape
        matrices[variable.name] = make_matrix(
            symbol.structure,
            symbol.property,
            base_shape[0],
            base_shape[1],
            rng,
        )

    return matrices


def _transpose_parity(operand: Operand) -> bool:
    parity = False
    while isinstance(operand, UnaryOp):
        if operand.op == "trans":
            parity = not parity
        operand = operand.operand
    return parity


def _contains_inverse(operand: Operand) -> bool:
    while isinstance(operand, UnaryOp):
        if operand.op == "inv":
            return True
        operand = operand.operand
    return False


def materialize_operand(operand: Operand, mats: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(operand, Var):
        return mats[operand.name]
    value = materialize_operand(operand.operand, mats)
    if operand.op == "trans":
        return value.T
    if operand.op == "inv":
        return np.linalg.inv(value)
    raise ExecuteError(f"unknown operand operator {operand.op!r}")


def reference_value(program: Program, mats: dict[str, np.ndarray]) -> np.ndarray:
    if not program.factors:
        raise ExecuteError("program has no factors")
    accumulator = materialize_operand(program.factors[0], mats)
    for factor in program.factors[1:]:
        accumulator = accumulator @ materialize_operand(factor, mats)
    return accumulator


def run_source(source: str, q: list[int], seed: int = 0) -> RunResult:
    checked = check_source(source)
    program = checked.program
    priced = price_source(source)
    if not priced:
        raise ExecuteError("compiler produced no variants")
    costs = [variant.eval(q) for variant in priced]
    generated_text = generate_source(source)
    rng = np.random.default_rng(seed)
    mats = build_mats(program, checked.symbols, q, rng)
    namespace = load_generated(generated_text)
    index = min(range(len(costs)), key=lambda item: costs[item])
    value = namespace["VARIANTS"][index](mats)
    reference = reference_value(program, mats)
    residual = frobenius_rel(value, reference)
    return RunResult(
        label=namespace["LABELS"][index],
        value=value,
        reference=reference,
        residual=residual,
        flops=costs[index],
        q=list(q),
    )


def load_generated(text: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "gmc_generated"}
    exec(compile(text, "<gmc-generated>", "exec"), namespace)
    return namespace


def frobenius_rel(a: np.ndarray, b: np.ndarray) -> float:
    numerator = np.linalg.norm(a - b)
    denominator = max(1.0, np.linalg.norm(b))
    return float(numerator / denominator)