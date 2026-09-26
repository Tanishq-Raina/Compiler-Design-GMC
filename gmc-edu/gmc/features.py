"""Matrix feature classification and product inference for GMC."""

from dataclasses import dataclass
from pathlib import Path

from gmc.ast import Operand, Program, UnaryOp, Var
from gmc.semantics import CheckedProgram, Symbol, check, check_file, check_source


class FeatureError(Exception):
    """An unsupported or invalid feature operation."""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


@dataclass(frozen=True)
class Feature:
    structure: str
    property: str


@dataclass
class Inference:
    checked: CheckedProgram
    factor_features: list[Feature]
    ltr_result: Feature


_STRUCTURE_CLASSES = {
    "Dense": "G",
    "Symmetric": "S",
    "Lower": "L",
    "UnitLower": "L",
    "Upper": "U",
    "UnitUpper": "U",
    "Diagonal": "D",
}
_PROPERTY_CLASSES = {
    "None": "SING",
    "FullRank": "INV",
    "SPD": "SPD",
    "Orthogonal": "ORTH",
}
_STRUCTURE_PRODUCTS = {
    "G": {"G": "G", "S": "G", "L": "G", "U": "G", "D": "G"},
    "S": {"G": "G", "S": "G", "L": "G", "U": "G", "D": "G"},
    "L": {"G": "G", "S": "G", "L": "L", "U": "G", "D": "L"},
    "U": {"G": "G", "S": "G", "L": "G", "U": "U", "D": "U"},
    "D": {"G": "G", "S": "G", "L": "L", "U": "U", "D": "D"},
}
_PROPERTY_PRODUCTS = {
    "SING": {"SING": "SING", "INV": "SING", "SPD": "SING", "ORTH": "SING"},
    "INV": {"SING": "SING", "INV": "INV", "SPD": "INV", "ORTH": "INV"},
    "SPD": {"SING": "SING", "INV": "INV", "SPD": "INV", "ORTH": "INV"},
    "ORTH": {"SING": "SING", "INV": "INV", "SPD": "INV", "ORTH": "ORTH"},
}


def structure_class(declared_structure: str) -> str:
    try:
        return _STRUCTURE_CLASSES[declared_structure]
    except KeyError as error:
        raise FeatureError(f"unknown structure class {declared_structure!r}") from error


def property_class(declared_property: str) -> str:
    try:
        return _PROPERTY_CLASSES[declared_property]
    except KeyError as error:
        raise FeatureError(f"unknown property class {declared_property!r}") from error


def feature_from_symbol(sym: Symbol) -> Feature:
    return Feature(
        structure_class(sym.structure),
        property_class(sym.property),
    )


def apply_transpose(feat: Feature) -> Feature:
    if feat.structure not in {"G", "S", "L", "U", "D"}:
        raise FeatureError(f"unknown structure class {feat.structure!r}")
    transposed = {"L": "U", "U": "L"}.get(feat.structure, feat.structure)
    return Feature(transposed, feat.property)


def apply_inv(feat: Feature) -> Feature:
    if feat.property == "SING":
        raise FeatureError("cannot invert a singular feature")
    if feat.property not in {"INV", "SPD", "ORTH"}:
        raise FeatureError(f"unknown property class {feat.property!r}")
    return feat


def feature_of_operand(operand: Operand, symbols: dict[str, Symbol]) -> Feature:
    if isinstance(operand, Var):
        try:
            symbol = symbols[operand.name]
        except KeyError as error:
            raise FeatureError(
                f"unknown symbol {operand.name!r}", operand.line, operand.column
            ) from error
        try:
            return feature_from_symbol(symbol)
        except FeatureError as error:
            raise FeatureError(str(error), operand.line, operand.column) from error

    feature = feature_of_operand(operand.operand, symbols)
    try:
        if operand.op == "trans":
            return apply_transpose(feature)
        if operand.op == "inv":
            return apply_inv(feature)
    except FeatureError as error:
        raise FeatureError(str(error), operand.line, operand.column) from error
    raise FeatureError(f"unknown unary operator {operand.op!r}", operand.line, operand.column)


def infer_structure(left: Feature, right: Feature) -> str:
    try:
        return _STRUCTURE_PRODUCTS[left.structure][right.structure]
    except KeyError as error:
        raise FeatureError(f"unknown structure class {error.args[0]!r}") from error


def infer_property(left: Feature, right: Feature) -> str:
    try:
        return _PROPERTY_PRODUCTS[left.property][right.property]
    except KeyError as error:
        raise FeatureError(f"unknown property class {error.args[0]!r}") from error


def infer_product(left: Feature, right: Feature) -> Feature:
    return Feature(
        infer_structure(left, right),
        infer_property(left, right),
    )


def infer_chain_ltr(features: list[Feature]) -> Feature:
    if not features:
        raise FeatureError("cannot infer an empty feature chain")
    result = features[0]
    for feature in features[1:]:
        result = infer_product(result, feature)
    return result


def infer(checked: CheckedProgram) -> Inference:
    factor_features = [
        feature_of_operand(factor, checked.symbols)
        for factor in checked.program.factors
    ]
    return Inference(
        checked=checked,
        factor_features=factor_features,
        ltr_result=infer_chain_ltr(factor_features),
    )


def infer_source(source: str) -> Inference:
    return infer(check_source(source))


def infer_file(path: str | Path) -> Inference:
    return infer(check_file(path))


def dump_inference(inf: Inference) -> str:
    lines = ["factors"]
    for factor, feature in zip(inf.checked.program.factors, inf.factor_features):
        lines.append(
            f"  {dump_operand(factor)} : {feature.structure}, {feature.property}"
        )
    lines.append(
        f"ltr_result : {inf.ltr_result.structure}, {inf.ltr_result.property}"
    )
    return "\n".join(lines)


def dump_operand(operand: Operand) -> str:
    if isinstance(operand, Var):
        return operand.name
    return f"{operand.op}({dump_operand(operand.operand)})"