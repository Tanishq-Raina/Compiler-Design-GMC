from pathlib import Path
import subprocess
import sys

import pytest

from gmc.ast import UnaryOp, Var
from gmc.parser import parse
from gmc.semantics import (
    SemanticError,
    base_var,
    check,
    check_file,
    check_source,
    dump_symbols,
    has_inv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_example(filename: str):
    return check_file(PROJECT_ROOT / "examples" / filename)


def test_simple_program_builds_symbols_in_declaration_order() -> None:
    checked = _check_example("simple.gmc")
    assert list(checked.symbols) == ["A", "L", "B"]
    assert (checked.symbols["A"].structure, checked.symbols["A"].property) == (
        "Dense",
        "None",
    )
    assert (checked.symbols["L"].structure, checked.symbols["L"].property) == (
        "Lower",
        "FullRank",
    )
    assert (checked.symbols["B"].structure, checked.symbols["B"].property) == (
        "Dense",
        "None",
    )


def test_nested_inverse_checks_legal_full_rank_property() -> None:
    checked = _check_example("nested.gmc")
    assert checked.symbols["A"].property == "FullRank"
    assert checked.symbols["L"].property == "FullRank"
    assert isinstance(checked.program.factors[0], UnaryOp)
    assert checked.program.factors[0].op == "inv"
    assert isinstance(checked.program.factors[0].operand, UnaryOp)
    assert checked.program.factors[0].operand.op == "trans"
    assert isinstance(checked.program.factors[0].operand.operand, Var)


def test_good_spd_program_and_symbol_dump() -> None:
    checked = _check_example("good_spd.gmc")
    assert (checked.symbols["A"].structure, checked.symbols["A"].property) == (
        "Symmetric",
        "SPD",
    )
    assert "A : Symmetric, SPD" in dump_symbols(checked)


def test_duplicate_declaration_points_at_second_declaration() -> None:
    with pytest.raises(SemanticError) as error:
        _check_example("dup_name.gmc")
    assert (error.value.line, error.value.column) == (2, 1)
    assert "duplicate" in str(error.value).lower()


def test_undeclared_name_points_at_variable() -> None:
    with pytest.raises(SemanticError) as error:
        _check_example("undeclared.gmc")
    assert (error.value.line, error.value.column) == (2, 5)
    assert "undeclared" in str(error.value).lower()


def test_illegal_combination_points_at_declaration() -> None:
    with pytest.raises(SemanticError) as error:
        _check_example("bad_combo.gmc")
    assert (error.value.line, error.value.column) == (1, 1)
    assert "structure" in str(error.value).lower()
    assert "property" in str(error.value).lower()


def test_inverse_of_noninvertible_matrix_points_at_inv() -> None:
    with pytest.raises(SemanticError) as error:
        _check_example("bad_inv.gmc")
    assert (error.value.line, error.value.column) == (3, 1)
    message = str(error.value)
    assert "inverse" in message.lower()
    assert "FullRank" in message
    assert "SPD" in message


def test_upper_orthogonal_is_an_illegal_combination() -> None:
    with pytest.raises(SemanticError):
        check_source("Matrix Q <Upper, Orthogonal>; Q")


def test_diagonal_spd_is_invertible() -> None:
    checked = check_source("Matrix Q <Diagonal, SPD>; inv(Q)")
    assert checked.symbols["Q"].property == "SPD"


def test_declaration_errors_take_priority_over_undeclared_names() -> None:
    source = "Matrix L <Lower, SPD>; L * Z"
    with pytest.raises(SemanticError) as error:
        check(parse(source))
    assert "combination" in str(error.value).lower()
    assert (error.value.line, error.value.column) == (1, 1)


def test_semantic_error_positions_are_integers() -> None:
    with pytest.raises(SemanticError) as error:
        check_source("Matrix A <Dense, None>; Z")
    assert isinstance(error.value.line, int)
    assert isinstance(error.value.column, int)


def test_ast_is_not_mutated_by_check() -> None:
    program = parse("Matrix A <Dense, FullRank>; inv(A)")
    original_factor = program.factors[0]
    checked = check(program)
    assert checked.program is program
    assert checked.program.factors[0] is original_factor
    assert has_inv(original_factor)
    assert base_var(original_factor).name == "A"


def test_check_cli_success_and_semantic_failure() -> None:
    good = subprocess.run(
        [sys.executable, "-m", "gmc", "check", "examples/simple.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0
    assert good.stdout.rstrip().endswith("OK")

    bad = subprocess.run(
        [sys.executable, "-m", "gmc", "check", "examples/undeclared.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert bad.stderr.startswith("semantic error at 2:5:")