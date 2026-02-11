"""CLI entry point: python -m context_engine "question" """

import argparse
import json
import sys

from context_engine.composer import compose
from context_engine.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_NUM_COLLAPSES,
    DEFAULT_TEMPERATURE,
)
from context_engine.context_constructor import construct_collapses
from context_engine.providers.openai_compat import OpenAICompatProvider


def main():
    parser = argparse.ArgumentParser(
        prog="context-engine",
        description="Reasoning through multiple independent collapses composed by necessity",
    )
    parser.add_argument("question", help="The question to reason about")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-collapses",
        type=int,
        default=DEFAULT_NUM_COLLAPSES,
        help=f"Number of independent collapses (default: {DEFAULT_NUM_COLLAPSES})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for collapse generation (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument("--save", metavar="FILE", help="Save full trace to JSON file")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print individual collapses before composition",
    )
    args = parser.parse_args()

    provider = OpenAICompatProvider(
        base_url=args.base_url,
        model=args.model,
    )

    # Construct collapses
    print(f"Generating {args.num_collapses} independent collapses...")
    try:
        collapses = construct_collapses(
            question=args.question,
            provider=provider,
            num_collapses=args.num_collapses,
            temperature=args.temperature,
        )
    except Exception as e:
        if "Connection" in type(e).__name__ or "connection" in str(e).lower():
            print(
                f"\nCould not connect to {args.base_url}\n"
                "Is vllm-mlx running? Start it with:\n"
                "  vllm-mlx serve ~/Models/Qwen3-8B-4bit --port 8000 --continuous-batching",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    if args.verbose:
        for i, c in enumerate(collapses, 1):
            print(f"\n{'=' * 60}")
            print(f"COLLAPSE {i}")
            print(f"{'=' * 60}")
            print(c["response"])

    # Compose
    print(f"\nComposing {len(collapses)} collapses...")
    result = compose(
        question=args.question,
        collapses=collapses,
        provider=provider,
    )

    # Output
    print(f"\n{'=' * 60}")
    print("RESOLUTION")
    print(f"{'=' * 60}")
    print(result["resolution"])
    print(f"\n{'=' * 60}")
    print("SHADOWS")
    print(f"{'=' * 60}")
    print(result["shadows"])

    # Save trace
    if args.save:
        trace = {
            "question": args.question,
            "collapses": collapses,
            "resolution": result["resolution"],
            "shadows": result["shadows"],
            "raw_composition": result["raw_composition"],
        }
        with open(args.save, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"\nTrace saved to {args.save}")


if __name__ == "__main__":
    main()
