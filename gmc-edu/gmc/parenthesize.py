"""Parenthesization trees and fanning-out variants for GMC chains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gmc.ast import Operand, Program
from gmc.features import dump_operand
from gmc.parser import parse, parse_file
from gmc.semantics import check, has_inv


@dataclass(frozen=True)
class Leaf:
    index: int


@dataclass(frozen=True)
class Mul:
    left: Node
    right: Node


Node = Leaf | Mul


@dataclass
class VariantPlan:
    n: int
    all_trees: list[Node]
    fanout: list[Node]
    essential: list[Node]
    square_factors: set[int]
    classes: list[frozenset[int]]


def n_leaves(node: Node) -> int:
    if isinstance(node, Leaf):
        return 1
    return n_leaves(node.left) + n_leaves(node.right)


def span(node: Node) -> tuple[int, int]:
    if isinstance(node, Leaf):
        return node.index, node.index
    left_start, _ = span(node.left)
    _, right_end = span(node.right)
    return left_start, right_end


def assert_chain(node: Node, n: int) -> None:
    indices: list[int] = []

    def visit(current: Node) -> None:
        if isinstance(current, Leaf):
            indices.append(current.index)
            return
        visit(current.left)
        visit(current.right)

    visit(node)
    assert n >= 1
    assert indices == list(range(n))


def left_to_right(i: int, j: int) -> Node:
    if i < 0 or j < i:
        raise ValueError("expected a non-empty range with 0 <= i <= j")
    result: Node = Leaf(i)
    for index in range(i + 1, j + 1):
        result = Mul(result, Leaf(index))
    return result


def right_to_left(i: int, j: int) -> Node:
    if i < 0 or j < i:
        raise ValueError("expected a non-empty range with 0 <= i <= j")
    result: Node = Leaf(j)
    for index in range(j - 1, i - 1, -1):
        result = Mul(Leaf(index), result)
    return result


def fanout_tree(h: int, n: int) -> Node:
    if n < 1 or h < 0 or h > n:
        raise ValueError("fanout requires n >= 1 and 0 <= h <= n")
    if h == 0:
        return left_to_right(0, n - 1)
    if h == n:
        return right_to_left(0, n - 1)

    prefix = right_to_left(0, h - 1)
    suffix = left_to_right(h, n - 1)
    return Mul(prefix, suffix)


def all_parenthesizations(n: int) -> list[Node]:
    if n < 1:
        raise ValueError("a chain must contain at least one factor")

    def build(start: int, end: int) -> list[Node]:
        if start == end:
            return [Leaf(start)]

        trees: list[Node] = []
        length = end - start + 1
        for right_size in range(1, length):
            split = end - right_size
            left_trees = build(start, split)
            right_trees = build(split + 1, end)
            trees.extend(
                Mul(left, right)
                for left in left_trees
                for right in right_trees
            )
        return trees

    return build(0, n - 1)


def fanout_set(n: int) -> list[Node]:
    trees: list[Node] = []
    seen: set[Node] = set()
    for h in range(n + 1):
        tree = fanout_tree(h, n)
        if tree not in seen:
            seen.add(tree)
            trees.append(tree)
    return trees


def size_classes(n: int, square_factors: set[int]) -> list[frozenset[int]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    parents = list(range(n + 1))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for factor_index in square_factors:
        if factor_index < 0 or factor_index >= n:
            raise ValueError(f"square factor index out of range: {factor_index}")
        left_root = find(factor_index)
        right_root = find(factor_index + 1)
        if left_root != right_root:
            parents[right_root] = left_root

    groups: dict[int, set[int]] = {}
    for index in range(n + 1):
        groups.setdefault(find(index), set()).add(index)
    return sorted(
        (frozenset(group) for group in groups.values()),
        key=min,
    )


def fanout_essential(n: int, square_factors: set[int]) -> list[Node]:
    trees: list[Node] = []
    seen: set[Node] = set()
    for equivalence_class in size_classes(n, square_factors):
        tree = fanout_tree(min(equivalence_class), n)
        if tree not in seen:
            seen.add(tree)
            trees.append(tree)
    return trees


def square_factors_of(program: Program) -> set[int]:
    return {
        index
        for index, factor in enumerate(program.factors)
        if has_inv(factor)
    }


def dump_tree(node: Node, factors: list[Operand | str]) -> str:
    if isinstance(node, Leaf):
        factor = factors[node.index]
        return factor if isinstance(factor, str) else dump_operand(factor)
    return f"({dump_tree(node.left, factors)} * {dump_tree(node.right, factors)})"


def plan_variants(program: Program) -> VariantPlan:
    n = len(program.factors)
    squares = square_factors_of(program)
    classes = size_classes(n, squares)
    return VariantPlan(
        n=n,
        all_trees=all_parenthesizations(n),
        fanout=fanout_set(n),
        essential=fanout_essential(n, squares),
        square_factors=squares,
        classes=classes,
    )


def plan_source(source: str) -> VariantPlan:
    program = parse(source)
    check(program)
    return plan_variants(program)


def plan_file(path: str | Path) -> VariantPlan:
    program = parse_file(path)
    check(program)
    return plan_variants(program)


def dump_plan(plan: VariantPlan, program: Program) -> str:
    factors = program.factors
    labels: dict[Node, int] = {}
    for h in range(plan.n + 1):
        labels.setdefault(fanout_tree(h, plan.n), h)

    lines = [f"n = {plan.n}", f"squares = {sorted(plan.square_factors)}"]
    lines.append("classes " + " ".join(
        "[" + ", ".join(str(index) for index in sorted(group)) + "]"
        for group in plan.classes
    ))
    lines.append(f"all ({len(plan.all_trees)})")
    lines.extend(f"  {dump_tree(tree, factors)}" for tree in plan.all_trees)
    lines.append(f"fanout ({len(plan.fanout)})")
    lines.extend(
        f"  E{labels[tree]} {dump_tree(tree, factors)}" for tree in plan.fanout
    )
    lines.append(f"essential ({len(plan.essential)})")
    lines.extend(
        f"  E{labels[tree]} {dump_tree(tree, factors)}" for tree in plan.essential
    )
    return "\n".join(lines)