import subprocess
import sys
from pathlib import Path

import pytest

from gmc.lexer import LexError, Lexer
from gmc.tokens import TokenType


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lex_example(filename: str):
    source = (PROJECT_ROOT / "examples" / filename).read_text(encoding="utf-8")
    return Lexer(source).tokenize()


def test_simple_token_types() -> None:
    tokens = _lex_example("simple.gmc")
    assert [token.type for token in tokens] == [
        TokenType.MATRIX,
        TokenType.IDENT,
        TokenType.LANGLE,
        TokenType.STRUCTURE,
        TokenType.COMMA,
        TokenType.PROPERTY,
        TokenType.RANGLE,
        TokenType.SEMICOLON,
        TokenType.MATRIX,
        TokenType.IDENT,
        TokenType.LANGLE,
        TokenType.STRUCTURE,
        TokenType.COMMA,
        TokenType.PROPERTY,
        TokenType.RANGLE,
        TokenType.SEMICOLON,
        TokenType.MATRIX,
        TokenType.IDENT,
        TokenType.LANGLE,
        TokenType.STRUCTURE,
        TokenType.COMMA,
        TokenType.PROPERTY,
        TokenType.RANGLE,
        TokenType.SEMICOLON,
        TokenType.IDENT,
        TokenType.STAR,
        TokenType.INV,
        TokenType.LPAREN,
        TokenType.IDENT,
        TokenType.RPAREN,
        TokenType.STAR,
        TokenType.TRANS,
        TokenType.LPAREN,
        TokenType.IDENT,
        TokenType.RPAREN,
        TokenType.EOF,
    ]


def test_simple_lexemes() -> None:
    tokens = _lex_example("simple.gmc")
    assert [token.lexeme for token in tokens] == [
        "Matrix", "A", "<", "Dense", ",", "None", ">", ";",
        "Matrix", "L", "<", "Lower", ",", "FullRank", ">", ";",
        "Matrix", "B", "<", "Dense", ",", "None", ">", ";",
        "A", "*", "inv", "(", "L", ")", "*", "trans", "(", "B", ")", "",
    ]


def test_comments_and_blank_lines_disappear() -> None:
    assert [token.type for token in _lex_example("with_comments.gmc")] == [
        token.type for token in _lex_example("simple.gmc")
    ]


def test_token_positions_are_one_based() -> None:
    tokens = _lex_example("simple.gmc")
    assert (tokens[0].type, tokens[0].line, tokens[0].column) == (
        TokenType.MATRIX,
        1,
        1,
    )
    assert (tokens[1].lexeme, tokens[1].line, tokens[1].column) == ("A", 1, 8)


def test_identifier_shapes() -> None:
    tokens = Lexer("A1 foo_bar").tokenize()
    assert [(token.type, token.lexeme) for token in tokens] == [
        (TokenType.IDENT, "A1"),
        (TokenType.IDENT, "foo_bar"),
        (TokenType.EOF, ""),
    ]


def test_keywords_are_case_sensitive() -> None:
    tokens = Lexer("None Dense Matrix inv trans matrix").tokenize()
    assert [token.type for token in tokens] == [
        TokenType.PROPERTY,
        TokenType.STRUCTURE,
        TokenType.MATRIX,
        TokenType.INV,
        TokenType.TRANS,
        TokenType.IDENT,
        TokenType.EOF,
    ]


def test_bad_character_reports_position() -> None:
    with pytest.raises(LexError) as error:
        _lex_example("bad_char.gmc")
    assert (error.value.line, error.value.column) == (1, 22)


def test_empty_input_has_only_eof() -> None:
    tokens = Lexer("").tokenize()
    assert [(token.type, token.lexeme) for token in tokens] == [(TokenType.EOF, "")]


def test_lex_cli_runs_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gmc", "lex", "examples/simple.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines()[0].startswith("MATRIX")
    assert result.stdout.splitlines()[-1].startswith("EOF")