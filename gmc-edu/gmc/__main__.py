"""Command-line entry point for the GMC educational compiler."""

import argparse
import sys
from pathlib import Path

from gmc.ast import dump_ast
from gmc.codegen import CodegenError, generate_file
from gmc.cost import CostError, dump_priced, parse_sizes, price_source
from gmc.execute import ExecuteError, run_source
from gmc.features import FeatureError, dump_inference, infer_file
from gmc.lexer import LexError, Lexer
from gmc.lower import LowerError, dump_lowered, lower_file
from gmc.parenthesize import dump_plan, plan_variants
from gmc.parser import ParseError, parse_file
from gmc.semantics import SemanticError, check_file, dump_symbols


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gmc",
        usage="python -m gmc {lex,parse,check,infer,variants,lower,cost,compile,run} <file>",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    lex_parser = commands.add_parser("lex", help="print tokens from a source file")
    lex_parser.add_argument("file", type=Path)
    parse_parser = commands.add_parser("parse", help="print the parsed AST")
    parse_parser.add_argument("file", type=Path)
    check_parser = commands.add_parser("check", help="check declarations and uses")
    check_parser.add_argument("file", type=Path)
    infer_parser = commands.add_parser("infer", help="infer matrix features")
    infer_parser.add_argument("file", type=Path)
    variants_parser = commands.add_parser(
        "variants", help="list parenthesization variants"
    )
    variants_parser.add_argument("file", type=Path)
    lower_parser = commands.add_parser("lower", help="lower essential variants")
    lower_parser.add_argument("file", type=Path)
    cost_parser = commands.add_parser("cost", help="price essential variants")
    cost_parser.add_argument("file", type=Path)
    cost_parser.add_argument("--sizes")
    compile_parser = commands.add_parser("compile", help="generate Python variants")
    compile_parser.add_argument("file", type=Path)
    compile_parser.add_argument("-o", "--output", type=Path)
    run_parser = commands.add_parser("run", help="run a generated GMC variant")
    run_parser.add_argument("file", type=Path)
    run_parser.add_argument("--sizes", required=True)
    run_parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    parser = _build_parser()
    arguments = parser.parse_args()

    if arguments.command == "run":
        try:
            source = arguments.file.read_text(encoding="utf-8")
            program = parse_file(arguments.file)
            sizes = parse_sizes(arguments.sizes, len(program.factors))
            result = run_source(source, sizes, arguments.seed)
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except CostError as error:
            print(f"cost error: {error}", file=sys.stderr)
            return 1
        except ExecuteError as error:
            print(f"run error: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"picked {result.label}  flops={result.flops}")
        print(f"shape={result.value.shape}")
        print(f"residual={result.residual:.2e}")
        if result.residual >= 1e-8:
            print("run error: residual too large", file=sys.stderr)
            return 1
        return 0

    if arguments.command == "compile":
        output_path = arguments.output or Path(f"{arguments.file.stem}_gen.py")
        try:
            generate_file(str(arguments.file), str(output_path))
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except (LowerError, CostError, CodegenError) as error:
            print(f"compile error: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"wrote {output_path.name}")
        return 0

    if arguments.command == "cost":
        try:
            source = arguments.file.read_text(encoding="utf-8")
            priced = price_source(source)
            sizes = None
            if arguments.sizes is not None:
                factor_count = len(priced[0].lowered.steps) + 1
                sizes = parse_sizes(arguments.sizes, factor_count)
            print(dump_priced(priced, sizes))
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except LowerError as error:
            print(f"lower error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except CostError as error:
            print(f"cost error: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        return 0

    if arguments.command == "lower":
        try:
            lowered = lower_file(arguments.file)
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except LowerError as error:
            print(f"lower error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(dump_lowered(lowered))
        return 0

    if arguments.command == "variants":
        try:
            checked = check_file(arguments.file)
            plan = plan_variants(checked.program)
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(dump_plan(plan, checked.program))
        return 0

    if arguments.command == "infer":
        try:
            inference = infer_file(arguments.file)
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except FeatureError as error:
            print(f"feature error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(dump_inference(inference))
        return 0

    if arguments.command == "check":
        try:
            checked = check_file(arguments.file)
        except LexError as error:
            print(f"lex error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except ParseError as error:
            print(f"parse error at {error.line}:{error.column}: {error}", file=sys.stderr)
            return 1
        except SemanticError as error:
            print(
                f"semantic error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(dump_symbols(checked))
        print("OK")
        return 0

    if arguments.command == "parse":
        try:
            program = parse_file(arguments.file)
        except (ParseError, LexError) as error:
            print(
                f"parse error at {error.line}:{error.column}: {error}",
                file=sys.stderr,
            )
            return 1
        except OSError as error:
            print(error, file=sys.stderr)
            return 1
        print(dump_ast(program))
        return 0

    try:
        source = arguments.file.read_text(encoding="utf-8")
        tokens = Lexer(source).tokenize()
    except (OSError, LexError) as error:
        print(error, file=sys.stderr)
        return 1

    for token in tokens:
        print(f"{token.type.name:<14} {token.lexeme!r:<12} {token.line}:{token.column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())