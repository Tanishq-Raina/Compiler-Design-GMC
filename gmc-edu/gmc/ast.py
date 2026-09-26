"""Abstract syntax tree nodes for the GMC language."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Declaration:
    name: str
    structure: str
    property: str
    line: int
    column: int


@dataclass
class Var:
    name: str
    line: int
    column: int


@dataclass
class UnaryOp:
    op: str
    operand: Var | UnaryOp
    line: int
    column: int


Operand = Var | UnaryOp


@dataclass
class Program:
    declarations: list[Declaration]
    factors: list[Operand]
    line: int
    column: int


def dump_ast(program: Program) -> str:
    lines = ["Program"]
    for declaration in program.declarations:
        lines.append(
            f"  Decl {declaration.name} "
            f"<{declaration.structure}, {declaration.property}>"
        )
    lines.append("  Expr")
    for factor in program.factors:
        _append_operand(factor, 2, lines)
    return "\n".join(lines)


def _append_operand(operand: Operand, depth: int, lines: list[str]) -> None:
    indentation = "  " * depth
    if isinstance(operand, Var):
        lines.append(f"{indentation}Var {operand.name}")
        return

    lines.append(f"{indentation}Unary {operand.op}")
    _append_operand(operand.operand, depth + 1, lines)