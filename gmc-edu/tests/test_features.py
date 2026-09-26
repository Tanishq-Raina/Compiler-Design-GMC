from pathlib import Path

import pytest

from gmc.ast import UnaryOp, Var
from gmc.features import (
    Feature,
    FeatureError,
    apply_inv,
    apply_transpose,
    dump_inference,
    feature_of_operand,
    infer_chain_ltr,
    infer_file,
    infer_product,
    infer_source,
    infer_structure,
    property_class,
    structure_class,
)
from gmc.semantics import Symbol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _infer_example(filename: str):
    return infer_file(PROJECT_ROOT / "examples" / filename)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("G", "G"), ("S", "S"), ("D", "D"), ("L", "U"), ("U", "L")],
)
def test_apply_transpose(source: str, expected: str) -> None:
    feature = apply_transpose(Feature(source, "INV"))
    assert feature == Feature(expected, "INV")


@pytest.mark.parametrize(
    "feature",
    [Feature("L", "INV"), Feature("S", "SPD"), Feature("G", "ORTH")],
)
def test_apply_inverse_preserves_invertible_feature(feature: Feature) -> None:
    assert apply_inv(feature) == feature


def test_apply_inverse_rejects_singular_feature() -> None:
    with pytest.raises(FeatureError):
        apply_inv(Feature("G", "SING"))


def test_declared_class_mappings() -> None:
    assert structure_class("UnitLower") == "L"
    assert structure_class("Upper") == "U"
    assert property_class("None") == "SING"
    assert property_class("FullRank") == "INV"
    with pytest.raises(FeatureError):
        structure_class("unknown")
    with pytest.raises(FeatureError):
        property_class("unknown")


def test_feature_of_operand_applies_unaries_innermost_first() -> None:
    symbols = {
        "L": Symbol("L", "Lower", "FullRank", 1, 1),
        "U": Symbol("U", "Upper", "FullRank", 1, 1),
    }
    lower = Var("L", 1, 1)
    transposed_lower = UnaryOp("trans", lower, 1, 1)
    inverse_lower = UnaryOp("inv", lower, 1, 1)
    nested_inverse = UnaryOp(
        "inv", UnaryOp("trans", Var("U", 1, 1), 1, 5), 1, 1
    )

    assert feature_of_operand(transposed_lower, symbols) == Feature("U", "INV")
    assert feature_of_operand(inverse_lower, symbols) == Feature("L", "INV")
    assert feature_of_operand(nested_inverse, symbols) == Feature("L", "INV")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("U", "U", "U"),
        ("L", "L", "L"),
        ("U", "L", "G"),
        ("L", "U", "G"),
        ("S", "S", "G"),
        ("G", "G", "G"),
        ("G", "S", "G"),
        ("G", "L", "G"),
        ("G", "U", "G"),
        ("G", "D", "G"),
    ],
)
def test_infer_structure_paper_cases(left: str, right: str, expected: str) -> None:
    assert infer_structure(Feature(left, "INV"), Feature(right, "INV")) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("D", "D", "D"),
        ("D", "L", "L"),
        ("L", "D", "L"),
        ("D", "U", "U"),
        ("U", "D", "U"),
        ("D", "S", "G"),
    ],
)
def test_infer_structure_diagonal_extensions(left: str, right: str, expected: str) -> None:
    assert infer_structure(Feature(left, "INV"), Feature(right, "INV")) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("SING", "INV", "SING"),
        ("INV", "INV", "INV"),
        ("SPD", "SPD", "INV"),
        ("ORTH", "ORTH", "ORTH"),
        ("ORTH", "SPD", "INV"),
        ("SPD", "SING", "SING"),
    ],
)
def test_infer_property_table(left: str, right: str, expected: str) -> None:
    assert infer_product(Feature("G", left), Feature("G", right)).property == expected


def test_simple_file_factor_features_and_dump() -> None:
    inference = _infer_example("simple.gmc")
    assert inference.factor_features == [
        Feature("G", "SING"),
        Feature("L", "INV"),
        Feature("G", "SING"),
    ]
    assert inference.ltr_result == Feature("G", "SING")
    assert dump_inference(inference) == (
        "factors\n"
        "  A : G, SING\n"
        "  inv(L) : L, INV\n"
        "  trans(B) : G, SING\n"
        "ltr_result : G, SING"
    )


def test_triangular_example_uses_left_to_right_preview() -> None:
    inference = _infer_example("infer_tri.gmc")
    assert [feature.structure for feature in inference.factor_features] == [
        "U",
        "U",
        "L",
    ]
    assert inference.ltr_result == Feature("G", "INV")


def test_property_example_uses_left_to_right_preview() -> None:
    inference = _infer_example("infer_props.gmc")
    assert inference.ltr_result == Feature("G", "INV")
    assert inference.factor_features[:2] == [
        Feature("G", "ORTH"),
        Feature("G", "ORTH"),
    ]


def test_good_spd_example_result() -> None:
    inference = _infer_example("good_spd.gmc")
    assert inference.factor_features == [Feature("S", "SPD"), Feature("G", "SING")]
    assert inference.ltr_result == Feature("G", "SING")


def test_nested_operand_dump_is_compact_and_deterministic() -> None:
    first = infer_source("Matrix A <Dense, FullRank>; inv(trans(A))")
    second = infer_source("Matrix A <Dense, FullRank>; inv(trans(A))")
    assert dump_inference(first) == dump_inference(second)
    assert "inv(trans(A)) : G, INV" in dump_inference(first)


def test_empty_left_to_right_chain_raises_feature_error() -> None:
    with pytest.raises(FeatureError):
        infer_chain_ltr([])


def test_infer_cli_prints_left_to_right_result() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "gmc", "infer", "examples/infer_tri.gmc"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "ltr_result : G, INV" in result.stdout