from pathlib import Path
import subprocess
import sys

import pytest

from gmc.ast import Var
from gmc.parenthesize import (
    Leaf,
    Mul,
    all_parenthesizations,
    assert_chain,
    dump_plan,
    dump_tree,
    fanout_essential,
    fanout_set,
    fanout_tree,
    left_to_right,
    n_leaves,
    plan_file,
    right_to_left,
    size_classes,
    span,
)
from gmc.parser import parse_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["A", "B", "C", "D", "E"]


def _dump(node, count: int = 5) -> str:
    return dump_tree(node, LABELS[:count])


def _leaf_order(node) -> list[int]:
    if isinstance(node, Leaf):
        return [node.index]
    return _leaf_order(node.left) + _leaf_order(node.right)


def test_directional_associations_and_single_leaf() -> None:
    assert left_to_right(0, 0) == Leaf(0)
    assert _dump(left_to_right(0, 2), 3) == "((A * B) * C)"
    assert _dump(right_to_left(0, 2), 3) == "(A * (B * C))"


def test_catalan_parenthesization_counts_and_order() -> None:
    assert all_parenthesizations(1) == [Leaf(0)]
    assert len(all_parenthesizations(2)) == 1
    assert _dump(all_parenthesizations(2)[0], 2) == "(A * B)"
    assert [_dump(tree, 3) for tree in all_parenthesizations(3)] == [
        "((A * B) * C)",
        "(A * (B * C))",
    ]
    assert len(all_parenthesizations(4)) == 5
    assert len(all_parenthesizations(5)) == 14


@pytest.mark.parametrize("n", range(1, 6))
def test_parenthesizations_cover_each_chain_once(n: int) -> None:
    for tree in all_parenthesizations(n):
        assert span(tree) == (0, n - 1)
        assert _leaf_order(tree) == list(range(n))
        assert_chain(tree, n)


def test_fanout_tree_constructs_each_boundary_variant() -> None:
    assert fanout_tree(0, 4) == left_to_right(0, 3)
    assert fanout_tree(4, 4) == right_to_left(0, 3)
    assert _dump(fanout_tree(2, 4), 4) == "((A * B) * (C * D))"
    assert _dump(fanout_tree(1, 4), 4) == "(A * ((B * C) * D))"
    assert _dump(fanout_tree(3, 4), 4) == "((A * (B * C)) * D)"


@pytest.mark.parametrize(("n", "expected"), [(1, 1), (2, 1), (3, 2), (4, 5)])
def test_fanout_sets_are_unique_full_parenthesizations(n: int, expected: int) -> None:
    variants = fanout_set(n)
    all_trees = set(all_parenthesizations(n))
    assert len(variants) == expected
    assert all(tree in all_trees for tree in variants)


def test_size_classes_unify_adjacent_sizes_for_square_factors() -> None:
    assert size_classes(3, set()) == [
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
    ]
    assert size_classes(3, {1}) == [
        frozenset({0}),
        frozenset({1, 2}),
        frozenset({3}),
    ]


def test_essential_fanout_uses_minimum_class_indices_and_deduplicates() -> None:
    trees = fanout_essential(3, {1})
    assert len(trees) == 2
    assert [_dump(tree, 3) for tree in trees] == [
        "((A * B) * C)",
        "(A * (B * C))",
    ]


def test_simple_program_variant_plan() -> None:
    plan = plan_file(PROJECT_ROOT / "examples" / "simple.gmc")
    factors = parse_file(PROJECT_ROOT / "examples" / "simple.gmc").factors
    assert plan.n == 3
    assert plan.square_factors == {1}
    assert (len(plan.all_trees), len(plan.fanout)) == (2, 2)
    assert {dump_tree(tree, factors) for tree in plan.fanout} == {
        "((A * inv(L)) * trans(B))",
        "(A * (inv(L) * trans(B)))",
    }
    assert dump_tree(left_to_right(0, 2), factors) == "((A * inv(L)) * trans(B))"
    assert dump_tree(right_to_left(0, 2), factors) == "(A * (inv(L) * trans(B)))"


def test_four_factor_plan_has_five_trees_in_each_set() -> None:
    plan = plan_file(PROJECT_ROOT / "examples" / "four.gmc")
    program = parse_file(PROJECT_ROOT / "examples" / "four.gmc")
    assert plan.n == 4
    assert plan.square_factors == set()
    assert len(plan.all_trees) == len(plan.fanout) == len(plan.essential) == 5
    output = dump_plan(plan, program)
    assert "(((A * B) * C) * D)" in output
    assert "(A * (B * (C * D)))" in output
    assert "((A * B) * (C * D))" in output


def test_spd_example_marks_inverse_factor_as_square() -> None:
    plan = plan_file(PROJECT_ROOT / "examples" / "good_spd.gmc")
    program = parse_file(PROJECT_ROOT / "examples" / "good_spd.gmc")
    assert plan.n == 2
    assert plan.square_factors == {0}
    assert len(plan.fanout) == 1
    assert dump_tree(plan.fanout[0], program.factors) == "(inv(A) * B)"


def test_plan_dump_is_deterministic() -> None:
    path = PROJECT_ROOT / "examples" / "simple.gmc"
    first_program = parse_file(path)
    second_program = parse_file(path)
    assert dump_plan(plan_file(path), first_program) == dump_plan(
        plan_file(path), second_program
    )


def test_variants_cli_outputs_counts() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gmc", "variants", "examples/four.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "n = 4" in result.stdout
    assert "all (5)" in result.stdout
    assert "fanout (5)" in result.stdout