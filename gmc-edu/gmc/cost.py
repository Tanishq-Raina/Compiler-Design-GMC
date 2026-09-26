"""Integer and symbolic kernel costs for lowered GMC variants."""

from dataclasses import dataclass
from collections.abc import Iterator

from gmc.lower import LoweredVariant, OpState, Step, lower_program
from gmc.parenthesize import Leaf, Mul, Node, fanout_tree, plan_variants
from gmc.parser import parse
from gmc.semantics import check


class CostError(Exception):
    """Invalid kernel cost or size input."""


@dataclass
class PricedStep:
    step: Step
    a: int
    b: int
    c: int
    symbolic: str


@dataclass
class PricedVariant:
    lowered: LoweredVariant
    priced_steps: list[PricedStep]
    symbolic_sum: str

    def eval(self, q: list[int]) -> int:
        expected_size_count = len(self.lowered.steps) + 2
        if len(q) != expected_size_count:
            raise CostError(f"expected {expected_size_count} sizes, got {len(q)}")
        if any(type(size) is not int or size <= 0 for size in q):
            raise CostError("sizes must be positive integers")
        return sum(
            flops(
                priced.step.kernel,
                q[priced.a],
                q[priced.c],
                q[priced.b],
                priced.step.left,
                priced.step.right,
            )
            for priced in self.priced_steps
        )


def leaf_span(index: int) -> tuple[int, int, int]:
    return index, index, index + 1


def node_span(node: Node) -> tuple[int, int]:
    if isinstance(node, Leaf):
        return node.index, node.index
    left_lo, left_hi = node_span(node.left)
    right_lo, right_hi = node_span(node.right)
    if left_hi + 1 != right_lo:
        raise CostError("multiplication tree spans are not contiguous")
    return left_lo, right_hi


def split_index(node: Mul) -> int:
    if node_span(node.left)[1] + 1 != node_span(node.right)[0]:
        raise CostError("multiplication tree spans are not contiguous")
    return node_span(node.right)[0]


def triplet(node: Mul) -> tuple[int, int, int]:
    lo, hi = node_span(node)
    return lo, split_index(node), hi + 1


def flops(kernel: str, m: int, n: int, k: int, left: OpState, right: OpState) -> int:
    if any(type(size) is not int or size <= 0 for size in (m, n, k)):
        raise CostError("dimensions must be positive integers")
    formulas = {
        "GEMM": lambda: 2 * m * k * n,
        "SYMM": lambda: _symmetric_multiply(m, n, left, right),
        "TRMM": lambda: _triangular_multiply(m, n, left, right),
        "SYSYMM": lambda: 2 * m * m * m,
        "TRSYMM": lambda: m * m * m,
        "TRTRMM": lambda: _triangular_multiply_pair(m, left, right),
        "TRSM": lambda: _triangular_solve(m, n, left, right),
        "TRSYSV": lambda: m * m * m,
        "TRTRSV": lambda: _triangular_solve_pair(m, left, right),
        "GEGESV": lambda: _general_solve(m, n, left, right, 2),
        "SYGESV": lambda: _general_solve(m, n, left, right, 1),
        "POGESV": lambda: _general_solve(m, n, left, right, 1),
        "GESYSV": lambda: (8 * m * m * m) // 3,
        "SYSYSV": lambda: (7 * m * m * m) // 3,
        "POSYSV": lambda: (7 * m * m * m) // 3,
        "SYTRSV": lambda: (7 * m * m * m) // 3,
        "GETRSV": lambda: 2 * m * m * m,
        "POTRSV": lambda: _positive_triangular_solve(m, left, right),
    }
    try:
        return formulas[kernel]()
    except KeyError as error:
        raise CostError(f"unknown kernel {kernel!r}") from error


def _symmetric_multiply(m: int, n: int, left: OpState, right: OpState) -> int:
    if left.feature.structure == "S":
        return 2 * m * m * n
    if right.feature.structure == "S":
        return 2 * m * n * n
    raise CostError("SYMM requires a symmetric operand")


def _triangular_multiply(m: int, n: int, left: OpState, right: OpState) -> int:
    if left.feature.structure in {"L", "U", "D"}:
        return m * m * n
    if right.feature.structure in {"L", "U", "D"}:
        return m * n * n
    raise CostError("TRMM requires a triangular operand")


def _triangular_multiply_pair(m: int, left: OpState, right: OpState) -> int:
    triangular = {"L", "U", "D"}
    if left.feature.structure not in triangular or right.feature.structure not in triangular:
        raise CostError("TRTRMM requires two triangular operands")
    multiplier = 1 if left.feature.structure == right.feature.structure else 2
    return (multiplier * m * m * m) // 3


def _triangular_solve(m: int, n: int, left: OpState, right: OpState) -> int:
    if left.inverted:
        return m * m * n
    if right.inverted:
        return m * n * n
    raise CostError("TRSM requires an inverted operand")


def _triangular_solve_pair(m: int, left: OpState, right: OpState) -> int:
    triangular = {"L", "U", "D"}
    if not left.inverted and not right.inverted:
        raise CostError("TRTRSV requires an inverted operand")
    inverted = left if left.inverted else right
    other = right if left.inverted else left
    if (
        inverted.feature.structure in triangular
        and other.feature.structure in triangular
        and inverted.feature.structure == other.feature.structure
    ):
        return (m * m * m) // 3
    return m * m * m


def _general_solve(
    m: int,
    n: int,
    left: OpState,
    right: OpState,
    cubic_numerator: int,
) -> int:
    if left.inverted:
        return (cubic_numerator * m * m * m) // 3 + 2 * m * m * n
    if right.inverted:
        return (cubic_numerator * n * n * n) // 3 + 2 * n * n * m
    raise CostError("general solve requires an inverted operand")


def _positive_triangular_solve(m: int, left: OpState, right: OpState) -> int:
    if (left.inverted and right.feature.structure == "L") or (
        right.inverted and left.feature.structure == "U"
    ):
        return (5 * m * m * m) // 3
    return (7 * m * m * m) // 3


def flops_symbolic(
    kernel: str, a: int, b: int, c: int, left: OpState, right: OpState
) -> str:
    qa, qb, qc = f"q{a}", f"q{b}", f"q{c}"
    square_a = f"{qa}*{qa}*{qa}"
    square_c = f"{qc}*{qc}*{qc}"
    formulas = {
        "GEMM": lambda: f"2*{qa}*{qb}*{qc}",
        "SYMM": lambda: _symbolic_symmetric_multiply(qa, qc, left, right),
        "TRMM": lambda: _symbolic_triangular_multiply(qa, qc, left, right),
        "SYSYMM": lambda: f"2*{square_a}",
        "TRSYMM": lambda: square_a,
        "TRTRMM": lambda: _symbolic_triangular_pair(square_a, left, right),
        "TRSM": lambda: _symbolic_triangular_solve(qa, qc, left, right),
        "TRSYSV": lambda: square_a,
        "TRTRSV": lambda: _symbolic_triangular_solve_pair(square_a, left, right),
        "GEGESV": lambda: _symbolic_general_solve(square_a, square_c, qa, qc, left, right, 2),
        "SYGESV": lambda: _symbolic_general_solve(square_a, square_c, qa, qc, left, right, 1),
        "POGESV": lambda: _symbolic_general_solve(square_a, square_c, qa, qc, left, right, 1),
        "GESYSV": lambda: f"(8*{square_a})//3",
        "SYSYSV": lambda: f"(7*{square_a})//3",
        "POSYSV": lambda: f"(7*{square_a})//3",
        "SYTRSV": lambda: f"(7*{square_a})//3",
        "GETRSV": lambda: f"2*{square_a}",
        "POTRSV": lambda: _symbolic_positive_triangular_solve(square_a, left, right),
    }
    try:
        return formulas[kernel]()
    except KeyError as error:
        raise CostError(f"unknown kernel {kernel!r}") from error


def _symbolic_symmetric_multiply(
    m: str, n: str, left: OpState, right: OpState
) -> str:
    if left.feature.structure == "S":
        return f"2*{m}*{m}*{n}"
    if right.feature.structure == "S":
        return f"2*{m}*{n}*{n}"
    raise CostError("SYMM requires a symmetric operand")


def _symbolic_triangular_multiply(
    m: str, n: str, left: OpState, right: OpState
) -> str:
    if left.feature.structure in {"L", "U", "D"}:
        return f"{m}*{m}*{n}"
    if right.feature.structure in {"L", "U", "D"}:
        return f"{m}*{n}*{n}"
    raise CostError("TRMM requires a triangular operand")


def _symbolic_divided(formula: str, divisor: int) -> str:
    return f"({formula})//{divisor}"


def _symbolic_triangular_pair(
    cube: str, left: OpState, right: OpState
) -> str:
    triangular = {"L", "U", "D"}
    if left.feature.structure not in triangular or right.feature.structure not in triangular:
        raise CostError("TRTRMM requires two triangular operands")
    multiplier = 1 if left.feature.structure == right.feature.structure else 2
    numerator = cube if multiplier == 1 else f"{multiplier}*{cube}"
    return _symbolic_divided(numerator, 3)


def _symbolic_triangular_solve(
    m: str, n: str, left: OpState, right: OpState
) -> str:
    if left.inverted:
        return f"{m}*{m}*{n}"
    if right.inverted:
        return f"{m}*{n}*{n}"
    raise CostError("TRSM requires an inverted operand")


def _symbolic_triangular_solve_pair(
    cube: str, left: OpState, right: OpState
) -> str:
    triangular = {"L", "U", "D"}
    if not left.inverted and not right.inverted:
        raise CostError("TRTRSV requires an inverted operand")
    inverted = left if left.inverted else right
    other = right if left.inverted else left
    if (
        inverted.feature.structure in triangular
        and other.feature.structure in triangular
        and inverted.feature.structure == other.feature.structure
    ):
        return _symbolic_divided(cube, 3)
    return cube


def _symbolic_general_solve(
    cube_m: str,
    cube_n: str,
    m: str,
    n: str,
    left: OpState,
    right: OpState,
    numerator: int,
) -> str:
    if left.inverted:
        cubic = cube_m if numerator == 1 else f"{numerator}*{cube_m}"
        return f"({cubic})//3+2*{m}*{m}*{n}"
    if right.inverted:
        cubic = cube_n if numerator == 1 else f"{numerator}*{cube_n}"
        return f"({cubic})//3+2*{n}*{n}*{m}"
    raise CostError("general solve requires an inverted operand")


def _symbolic_positive_triangular_solve(
    cube: str, left: OpState, right: OpState
) -> str:
    if (left.inverted and right.feature.structure == "L") or (
        right.inverted and left.feature.structure == "U"
    ):
        return _symbolic_divided(f"5*{cube}", 3)
    return _symbolic_divided(f"7*{cube}", 3)


def _postorder_triplets(node: Node) -> Iterator[tuple[int, int, int]]:
    if isinstance(node, Leaf):
        return
    yield from _postorder_triplets(node.left)
    yield from _postorder_triplets(node.right)
    yield triplet(node)


def price_variant(tree: Node, lowered: LoweredVariant, n: int) -> PricedVariant:
    if n < 1:
        raise CostError("a priced chain must contain at least one factor")
    if node_span(tree) != (0, n - 1):
        raise CostError("tree does not span the requested factor chain")
    triples = list(_postorder_triplets(tree))
    if len(triples) != len(lowered.steps):
        raise CostError("tree multiplication count does not match lowered steps")

    priced_steps: list[PricedStep] = []
    for step, (a, b, c) in zip(lowered.steps, triples):
        if a < 0 or b < 0 or c > n:
            raise CostError("tree triplet is outside the chain dimensions")
        priced_steps.append(
            PricedStep(
                step=step,
                a=a,
                b=b,
                c=c,
                symbolic=flops_symbolic(step.kernel, a, b, c, step.left, step.right),
            )
        )
    symbolic_sum = "+".join(item.symbolic for item in priced_steps) or "0"
    return PricedVariant(lowered, priced_steps, symbolic_sum)


def price_source(source: str) -> list[PricedVariant]:
    program = parse(source)
    checked = check(program)
    variant_plan = plan_variants(program)
    labels: dict[Node, str] = {}
    for h in range(variant_plan.n + 1):
        labels.setdefault(fanout_tree(h, variant_plan.n), f"E{h}")
    trees_with_labels = [
        (labels[tree], tree) for tree in variant_plan.essential
    ]
    lowered_plan = lower_program(program, checked.symbols, trees_with_labels)
    return [
        price_variant(tree, lowered, variant_plan.n)
        for tree, lowered in zip(variant_plan.essential, lowered_plan.variants)
    ]


def parse_sizes(value: str, n: int) -> list[int]:
    try:
        sizes = [] if not value.strip() else [
            int(part.strip()) for part in value.split(",")
        ]
    except ValueError as error:
        raise CostError("sizes must be comma-separated integers") from error
    expected = n + 1
    if len(sizes) != expected:
        raise CostError(f"expected {expected} sizes, got {len(sizes)}")
    if any(size <= 0 for size in sizes):
        raise CostError("sizes must be positive integers")
    return sizes


def pick_best(priced: list[PricedVariant], q: list[int]) -> int:
    if not priced:
        raise CostError("cannot select a best variant from an empty list")
    return min(range(len(priced)), key=lambda index: priced[index].eval(q))


def dump_priced(priced: list[PricedVariant], q: list[int] | None) -> str:
    lines: list[str] = []
    for variant in priced:
        lines.append(f"variant {variant.lowered.h_label} {variant.lowered.tree_dump}")
        lines.extend(
            f"  {item.step.dest} = {item.step.kernel} "
            f"{item.step.left.source}, {item.step.right.source}  # {item.symbolic}"
            for item in variant.priced_steps
        )
        lines.append(f"    cost = {variant.symbolic_sum}")
        if q is not None:
            lines.append(f"    flops = {variant.eval(q)}")

    if q is not None and priced:
        best_index = pick_best(priced, q)
        best = priced[best_index]
        lines.append(
            f"best = {best.lowered.h_label} (flops = {best.eval(q)})"
        )
    return "\n".join(lines)