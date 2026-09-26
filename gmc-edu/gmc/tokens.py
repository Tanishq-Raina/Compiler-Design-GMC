"""Token definitions for the GMC language."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    MATRIX = auto()
    INV = auto()
    TRANS = auto()
    STRUCTURE = auto()
    PROPERTY = auto()
    IDENT = auto()
    LANGLE = auto()
    RANGLE = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    SEMICOLON = auto()
    STAR = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str
    line: int
    column: int