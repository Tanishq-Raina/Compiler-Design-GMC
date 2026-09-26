"""Lower essential GMC parenthesizations to BLAS-like kernel steps."""

from dataclasses import dataclass, replace
from pathlib import Path

from gmc.ast import Program
from gmc.features import (
    Feature,
    FeatureError,
    apply_inv,
    dump_operand,
    feature_of_operand,
    infer_product,
)
from gmc.parenthesize import (
    Leaf,
    Mul,
    Node,
    VariantPlan,
    dump_tree,
    fanout_tree,
    plan_variants,
)
from gmc.parser import parse_file
from gmc.semantics import (
    CheckedProgram,
    Symbol,
    check_file,
    check_source,
    has_inv,
)


class LowerError(Exception):
    """An unsupported lower-level operation with an optional source position."""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


@dataclass(frozen=True)
class OpState:
    name: str
    feature: Feature
    inverted: bool
    source: str


@dataclass
class Step:
    kernel: str
    dest: str
    left: OpState
    right: OpState
    result_inverted: bool
    result_feature: Feature


@dataclass
class LoweredVariant:
    h_label: str
    tree_dump: str
    steps: list[Step]
    result: OpState


@dataclass
class LoweredPlan:
    variants: list[LoweredVariant]


_GENERAL_STRUCTURES = {"G", "S"}
_TRIANGULAR_STRUCTURES = {"L", "U", "D"}


def both_inverted(a: OpState, b: OpState) -> bool:
    return a.inverted and b.inverted


def should_propagate(a: OpState, b: OpState) -> bool:
    if a.inverted == b.inverted:
        return False
    inverted_side, other_side = (a, b) if a.inverted else (b, a)
    return (
        inverted_side.feature.structure in _GENERAL_STRUCTURES
        and (
            other_side.feature.structure in {"L", "U"}
            or other_side.feature.property == "ORTH"
        )
    )


def rewrite_association(
    a: OpState, b: OpState
) -> tuple[OpState, OpState, bool]:
    if both_inverted(a, b):
        return replace(b, inverted=False), replace(a, inverted=False), True

    if not should_propagate(a, b):
        return a, b, False

    if a.inverted:
        return replace(b, inverted=True), replace(a, inverted=False), True
    return replace(b, inverted=False), replace(a, inverted=True), True


def assign_kernel(left: OpState, right: OpState) -> str:
    if left.inverted and right.inverted:
        raise LowerError("kernel assignment requires at most one inverted operand")
    left_structure = _kernel_structure(left.feature.structure)
    right_structure = _kernel_structure(right.feature.structure)

    if not left.inverted and not right.inverted:
        return _multiply_kernel(left_structure, right_structure)

    inverted = left if left.inverted else right
    other_structure = right_structure if left.inverted else left_structure
    inverted_structure = _kernel_structure(inverted.feature.structure)

    if inverted_structure == "G":
        kernels = {"G": "GEGESV", "S": "GESYSV", "L": "GETRSV"}
        return _lookup_kernel(kernels, other_structure)
    if inverted_structure == "S":
        if inverted.feature.property == "SPD":
            kernels = {"G": "POGESV", "S": "POSYSV", "L": "POTRSV"}
        else:
            kernels = {"G": "SYGESV", "S": "SYSYSV", "L": "SYTRSV"}
        return _lookup_kernel(kernels, other_structure)
    if inverted_structure in {"L", "U"}:
        kernels = {"G": "TRSM", "S": "TRSYSV", "L": "TRTRSV"}
        return _lookup_kernel(kernels, other_structure)
    raise LowerError(f"unsupported inverted structure {inverted_structure!r}")


def _kernel_structure(structure: str) -> str:
    if structure not in {"G", "S", "L", "U", "D"}:
        raise LowerError(f"unknown structure class {structure!r}")
    return "L" if structure == "D" else structure


def _lookup_kernel(kernels: dict[str, str], structure: str) -> str:
    try:
        return kernels[structure]
    except KeyError as error:
        raise LowerError(
            f"unsupported kernel combination with structure {structure!r}"
        ) from error


def _multiply_kernel(left: str, right: str) -> str:
    triangular_left = left in {"L", "U"}
    triangular_right = right in {"L", "U"}
    if left == "G" and right == "G":
        return "GEMM"
    if left == "S" and right == "S":
        return "SYSYMM"
    if (left == "S" and right == "G") or (left == "G" and right == "S"):
        return "SYMM"
    if triangular_left and triangular_right:
        return "TRTRMM"
    if left == "S" and triangular_right or triangular_left and right == "S":
        return "TRSYMM"
    if triangular_left and right == "G" or left == "G" and triangular_right:
        return "TRMM"
    raise LowerError(
        f"unsupported multiply combination {left!r} by {right!r}"
    )


def lower_tree(
    tree: Node, program: Program, symbols: dict[str, Symbol]
) -> list[Step]:
    steps: list[Step] = []
    _lower_node(tree, program, symbols, steps)
    return steps


def _lower_node(
    node: Node,
    program: Program,
    symbols: dict[str, Symbol],
    steps: list[Step],
) -> OpState:
    if isinstance(node, Leaf):
        if node.index < 0 or node.index >= len(program.factors):
            raise LowerError(f"leaf index out of range: {node.index}")
        try:
            operand = program.factors[node.index]
        except IndexError as error:
            raise LowerError(f"leaf index out of range: {node.index}") from error
        source = dump_operand(operand)
        return OpState(
            name=source,
            feature=feature_of_operand(operand, symbols),
            inverted=has_inv(operand),
            source=source,
        )

    left_state = _lower_node(node.left, program, symbols, steps)
    right_state = _lower_node(node.right, program, symbols, steps)
    left, right, result_inverted = rewrite_association(left_state, right_state)
    kernel = assign_kernel(left, right)
    try:
        result_feature = infer_product(left.feature, right.feature)
        if result_inverted:
            result_feature = apply_inv(result_feature)
    except FeatureError as error:
        raise LowerError(
            f"could not infer lowered result feature: {error}",
            error.line,
            error.column,
        ) from error

    destination = f"T{len(steps) + 1}"
    steps.append(
        Step(
            kernel=kernel,
            dest=destination,
            left=left,
            right=right,
            result_inverted=result_inverted,
            result_feature=result_feature,
        )
    )
    return OpState(
        name=destination,
        feature=result_feature,
        inverted=result_inverted,
        source=destination,
    )


def lower_program(
    program: Program,
    symbols: dict[str, Symbol],
    trees_with_labels: list[tuple[str, Node]],
) -> LoweredPlan:
    variants: list[LoweredVariant] = []
    for label, tree in trees_with_labels:
        steps: list[Step] = []
        result = _lower_node(tree, program, symbols, steps)
        variants.append(
            LoweredVariant(
                h_label=label,
                tree_dump=dump_tree(tree, program.factors),
                steps=steps,
                result=result,
            )
        )
    return LoweredPlan(variants)


def lower_source(source: str) -> LoweredPlan:
    checked = check_source(source)
    return _lower_checked_program(checked)


def lower_file(path: str | Path) -> LoweredPlan:
    checked = check_file(path)
    return _lower_checked_program(checked)


def _lower_checked_program(checked: CheckedProgram) -> LoweredPlan:
    variant_plan: VariantPlan = plan_variants(checked.program)
    labels: dict[Node, str] = {}
    for h in range(variant_plan.n + 1):
        labels.setdefault(fanout_tree(h, variant_plan.n), f"E{h}")
    trees_with_labels = [
        (labels[tree], tree) for tree in variant_plan.essential
    ]
    return lower_program(checked.program, checked.symbols, trees_with_labels)


def dump_lowered(plan: LoweredPlan) -> str:
    lines: list[str] = []
    for variant in plan.variants:
        lines.append(f"variant {variant.h_label} {variant.tree_dump}")
        lines.extend(
            f"  {step.dest} = {step.kernel} {step.left.source}, {step.right.source}"
            for step in variant.steps
        )
    return "\n".join(lines)