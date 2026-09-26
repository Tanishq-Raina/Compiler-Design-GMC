from pathlib import Path

import pytest

from gmc.ast import Program, UnaryOp, Var, dump_ast
from gmc.lexer import LexError
from gmc.parser import ParseError, parse, parse_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_example(filename: str) -> Program:
    return parse_file(PROJECT_ROOT / "examples" / filename)


def _operand_shape(operand: Var | UnaryOp) -> tuple:
    if isinstance(operand, Var):
        return ("Var", operand.name)
    return ("Unary", operand.op, _operand_shape(operand.operand))


def _program_shape(program: Program) -> tuple:
    declarations = tuple(
        (item.name, item.structure, item.property) for item in program.declarations
    )
    factors = tuple(_operand_shape(factor) for factor in program.factors)
    return declarations, factors


def test_simple_program_has_flat_factors_and_expected_dump() -> None:
    program = _parse_example("simple.gmc")
    output = dump_ast(program)

    for expected in (
        "Decl A <Dense, None>",
        "Decl L <Lower, FullRank>",
        "Decl B <Dense, None>",
        "Var A",
        "Unary inv",
        "Var L",
        "Unary trans",
        "Var B",
    ):
        assert expected in output
    assert len(program.factors) == 3


def test_comments_example_has_equivalent_ast() -> None:
    assert _program_shape(_parse_example("with_comments.gmc")) == _program_shape(
        _parse_example("simple.gmc")
    )


def test_nested_unary_operations_keep_their_tree_shape() -> None:
    program = _parse_example("nested.gmc")
    first, second, third = program.factors

    assert isinstance(first, UnaryOp)
    assert first.op == "inv"
    assert isinstance(first.operand, UnaryOp)
    assert first.operand.op == "trans"
    assert isinstance(first.operand.operand, Var)
    assert first.operand.operand.name == "A"

    assert isinstance(second, Var)
    assert second.name == "L"

    assert isinstance(third, UnaryOp)
    assert third.op == "trans"
    assert isinstance(third.operand, UnaryOp)
    assert third.operand.op == "inv"
    assert isinstance(third.operand.operand, Var)
    assert third.operand.operand.name == "A"
    assert program.declarations[1].property == "FullRank"


def test_program_without_declarations_is_legal() -> None:
    program = parse("X * trans(Y)")
    assert program.declarations == []
    assert len(program.factors) == 2


def test_missing_declaration_semicolon_is_a_parse_error() -> None:
    with pytest.raises(ParseError):
        _parse_example("syntax_bad.gmc")


def test_extra_token_after_expression_is_a_parse_error() -> None:
    with pytest.raises(ParseError):
        parse("Matrix A <Dense, None>; A * B ;")


def test_structure_and_property_slots_are_checked() -> None:
    with pytest.raises(ParseError):
        parse("Matrix A <FullRank, Lower>;")


@pytest.mark.parametrize(
    "source",
    ["Matrix A <Dense, None; A", "Matrix A <Dense None>; A"],
)
def test_missing_angle_bracket_or_comma_is_a_parse_error(source: str) -> None:
    with pytest.raises(ParseError):
        parse(source)


def test_empty_input_needs_an_expression() -> None:
    with pytest.raises(ParseError):
        parse("")


def test_lexer_and_parser_errors_have_integer_positions() -> None:
    with pytest.raises(ParseError) as parse_error:
        parse("A *")
    assert isinstance(parse_error.value.line, int)
    assert isinstance(parse_error.value.column, int)

    with pytest.raises(LexError) as lex_error:
        parse("@")
    assert isinstance(lex_error.value.line, int)
    assert isinstance(lex_error.value.column, int)


def test_ast_dump_is_deterministic() -> None:
    assert dump_ast(_parse_example("simple.gmc")) == dump_ast(
        _parse_example("simple.gmc")
    )


def test_parse_cli_prints_ast() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "gmc", "parse", "examples/simple.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("Program\n")
    assert "Unary inv" in result.stdout


@pytest.mark.parametrize(
    ("filename", "location"),
    [("syntax_bad.gmc", "2:1"), ("bad_char.gmc", "1:22")],
)
def test_parse_cli_reports_positioned_errors(filename: str, location: str) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "gmc", "parse", f"examples/{filename}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr.startswith(f"parse error at {location}:")