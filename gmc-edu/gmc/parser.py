"""Recursive-descent parser for the Phase 2 GMC language."""

from pathlib import Path

from gmc.ast import Declaration, Operand, Program, UnaryOp, Var
from gmc.lexer import Lexer
from gmc.tokens import Token, TokenType


class ParseError(Exception):
    """A syntax error with its 1-based source position."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


class Parser:
    def __init__(self, source: str) -> None:
        self._lexer = Lexer(source)
        self._current = self._lexer.next_token()

    def parse(self) -> Program:
        declarations: list[Declaration] = []
        while self.peek().type is TokenType.MATRIX:
            declarations.append(self._parse_declaration())

        expression_start = self.peek()
        factors = [self._parse_operand()]
        while self.accept(TokenType.STAR) is not None:
            factors.append(self._parse_operand())
        self.expect(TokenType.EOF)
        return Program(
            declarations=declarations,
            factors=factors,
            line=expression_start.line,
            column=expression_start.column,
        )

    def peek(self) -> Token:
        return self._current

    def advance(self) -> Token:
        token = self._current
        if token.type is not TokenType.EOF:
            self._current = self._lexer.next_token()
        return token

    def accept(self, token_type: TokenType) -> Token | None:
        if self.peek().type is token_type:
            return self.advance()
        return None

    def expect(self, token_type: TokenType) -> Token:
        token = self.peek()
        if token.type is not token_type:
            found = "end of input" if token.type is TokenType.EOF else (
                f"{token.type.name} {token.lexeme!r}"
            )
            raise ParseError(
                f"expected {token_type.name}, found {found}",
                token.line,
                token.column,
            )
        return self.advance()

    def _parse_declaration(self) -> Declaration:
        declaration_start = self.expect(TokenType.MATRIX)
        name = self.expect(TokenType.IDENT)
        self.expect(TokenType.LANGLE)
        structure = self.expect(TokenType.STRUCTURE)
        self.expect(TokenType.COMMA)
        property_token = self.expect(TokenType.PROPERTY)
        self.expect(TokenType.RANGLE)
        self.expect(TokenType.SEMICOLON)
        return Declaration(
            name=name.lexeme,
            structure=structure.lexeme,
            property=property_token.lexeme,
            line=declaration_start.line,
            column=declaration_start.column,
        )

    def _parse_operand(self) -> Operand:
        token = self.peek()
        if token.type is TokenType.IDENT:
            self.advance()
            return Var(token.lexeme, token.line, token.column)

        if token.type in (TokenType.INV, TokenType.TRANS):
            operator = self.advance()
            self.expect(TokenType.LPAREN)
            operand = self._parse_operand()
            self.expect(TokenType.RPAREN)
            return UnaryOp(operator.lexeme, operand, operator.line, operator.column)

        found = "end of input" if token.type is TokenType.EOF else (
            f"{token.type.name} {token.lexeme!r}"
        )
        raise ParseError(
            f"expected operand (IDENT, INV, or TRANS), found {found}",
            token.line,
            token.column,
        )


def parse(source: str) -> Program:
    return Parser(source).parse()


def parse_file(path: str | Path) -> Program:
    return parse(Path(path).read_text(encoding="utf-8"))