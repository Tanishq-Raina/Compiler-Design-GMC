"""Symbol and declaration-use checks for GMC programs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from gmc.ast import Operand, Program, UnaryOp, Var
from gmc.parser import parse, parse_file


class SemanticError(Exception):
    """A semantic error with its 1-based source position."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


@dataclass(frozen=True)
class Symbol:
    name: str
    structure: str
    property: str
    line: int
    column: int


@dataclass
class CheckedProgram:
    program: Program
    symbols: dict[str, Symbol]


_ALLOWED_PROPERTIES = {
    "Dense": {"None", "FullRank", "SPD", "Orthogonal"},
    "Symmetric": {"None", "FullRank", "SPD", "Orthogonal"},
    "Lower": {"None", "FullRank"},
    "UnitLower": {"None", "FullRank"},
    "Upper": {"None", "FullRank"},
    "UnitUpper": {"None", "FullRank"},
    "Diagonal": {"None", "FullRank", "SPD"},
}
_INVERTIBLE_PROPERTIES = {"FullRank", "SPD", "Orthogonal"}


def check(program: Program) -> CheckedProgram:
    for declaration in program.declarations:
        allowed = _ALLOWED_PROPERTIES.get(declaration.structure, set())
        if declaration.property not in allowed:
            raise SemanticError(
                "illegal structure/property combination: "
                f"{declaration.structure} with {declaration.property}",
                declaration.line,
                declaration.column,
            )

    symbols: dict[str, Symbol] = {}
    for declaration in program.declarations:
        if declaration.name in symbols:
            raise SemanticError(
                f"duplicate declaration: {declaration.name} is already declared",
                declaration.line,
                declaration.column,
            )
        symbols[declaration.name] = Symbol(
            name=declaration.name,
            structure=declaration.structure,
            property=declaration.property,
            line=declaration.line,
            column=declaration.column,
        )

    for factor in program.factors:
        for variable in _iter_vars(factor):
            if variable.name not in symbols:
                raise SemanticError(
                    f"undeclared identifier {variable.name!r}",
                    variable.line,
                    variable.column,
                )

    for factor in program.factors:
        for unary in iter_unaries(factor):
            if unary.op == "inv":
                variable = base_var(unary.operand)
                symbol = symbols[variable.name]
                if symbol.property not in _INVERTIBLE_PROPERTIES:
                    raise SemanticError(
                        "inverse requires a FullRank, SPD, or Orthogonal property; "
                        f"{variable.name} has {symbol.property}",
                        unary.line,
                        unary.column,
                    )

    return CheckedProgram(program=program, symbols=symbols)


def check_source(source: str) -> CheckedProgram:
    return check(parse(source))


def check_file(path: str | Path) -> CheckedProgram:
    return check(parse_file(path))


def dump_symbols(checked: CheckedProgram) -> str:
    return "\n".join(
        f"{symbol.name} : {symbol.structure}, {symbol.property}"
        for symbol in checked.symbols.values()
    )


def base_var(operand: Operand) -> Var:
    while isinstance(operand, UnaryOp):
        operand = operand.operand
    return operand


def has_inv(operand: Operand) -> bool:
    return any(unary.op == "inv" for unary in iter_unaries(operand))


def iter_unaries(operand: Operand) -> Iterator[UnaryOp]:
    while isinstance(operand, UnaryOp):
        yield operand
        operand = operand.operand


def _iter_vars(operand: Operand) -> Iterator[Var]:
    while isinstance(operand, UnaryOp):
        operand = operand.operand
    yield operand