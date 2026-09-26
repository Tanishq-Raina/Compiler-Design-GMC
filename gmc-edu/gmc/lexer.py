"""Hand-written lexer for the Phase 1 GMC language."""

from dataclasses import dataclass

from gmc.tokens import Token, TokenType


class LexError(Exception):
    """An unrecognized character with its 1-based source position."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


@dataclass(frozen=True)
class _Keyword:
    type: TokenType


_KEYWORDS = {
    "Matrix": _Keyword(TokenType.MATRIX),
    "inv": _Keyword(TokenType.INV),
    "trans": _Keyword(TokenType.TRANS),
    "Dense": _Keyword(TokenType.STRUCTURE),
    "Symmetric": _Keyword(TokenType.STRUCTURE),
    "Lower": _Keyword(TokenType.STRUCTURE),
    "UnitLower": _Keyword(TokenType.STRUCTURE),
    "Upper": _Keyword(TokenType.STRUCTURE),
    "UnitUpper": _Keyword(TokenType.STRUCTURE),
    "Diagonal": _Keyword(TokenType.STRUCTURE),
    "None": _Keyword(TokenType.PROPERTY),
    "FullRank": _Keyword(TokenType.PROPERTY),
    "SPD": _Keyword(TokenType.PROPERTY),
    "Orthogonal": _Keyword(TokenType.PROPERTY),
}

_SINGLE_CHAR_TOKENS = {
    "<": TokenType.LANGLE,
    ">": TokenType.RANGLE,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
    ";": TokenType.SEMICOLON,
    "*": TokenType.STAR,
}


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self._index = 0
        self._line = 1
        self._column = 1

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type is TokenType.EOF:
                return tokens

    def next_token(self) -> Token:
        self._skip_ignored()
        line, column = self._line, self._column

        if self._at_end():
            return Token(TokenType.EOF, "", line, column)

        character = self._current()
        if self._is_identifier_start(character):
            return self._read_identifier(line, column)

        token_type = _SINGLE_CHAR_TOKENS.get(character)
        if token_type is not None:
            self._advance()
            return Token(token_type, character, line, column)

        raise LexError(f"unexpected character {character!r}", line, column)

    def _skip_ignored(self) -> None:
        while not self._at_end():
            character = self._current()
            if character in " \t\n\r":
                self._advance()
            elif character == "#":
                while not self._at_end() and self._current() not in "\n\r":
                    self._advance()
            else:
                return

    def _read_identifier(self, line: int, column: int) -> Token:
        start = self._index
        self._advance()
        while not self._at_end() and self._is_identifier_part(self._current()):
            self._advance()

        lexeme = self.source[start : self._index]
        keyword = _KEYWORDS.get(lexeme)
        token_type = keyword.type if keyword is not None else TokenType.IDENT
        return Token(token_type, lexeme, line, column)

    def _advance(self) -> None:
        character = self.source[self._index]
        self._index += 1
        if character == "\n" or (character == "\r" and self._current_or_empty() != "\n"):
            self._line += 1
            self._column = 1
        else:
            self._column += 1

    def _current(self) -> str:
        return self.source[self._index]

    def _current_or_empty(self) -> str:
        if self._at_end():
            return ""
        return self.source[self._index]

    def _at_end(self) -> bool:
        return self._index >= len(self.source)

    @staticmethod
    def _is_identifier_start(character: str) -> bool:
        return character.isascii() and (character.isalpha() or character == "_")

    @staticmethod
    def _is_identifier_part(character: str) -> bool:
        return character.isascii() and (character.isalnum() or character == "_")