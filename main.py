"""Entry point for all benchmark scripts.

Usage:
    uv run python -m main --script benchmarks.baseline --model qwen-0.5b
    uv run python -m main --script benchmarks.baseline --model qwen-0.5b --n-prompts 5
    uv run python -m main --script benchmarks.baseline --model llama-1b-cuda
"""

import argparse
import asyncio
import importlib

from config import resolve_model

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Serving Lab benchmark runner")
    parser.add_argument(
        "--script",
        required=True,
        help="Dotted module path to benchmark script e.g. benchmarks.baseline",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model alias from registry (e.g. qwen-0.5b) or full HuggingFace name",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps"],
        default=None,
        help="Override device from model registry",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=None,
        help="Limit number of prompts to send (useful for quick smoke tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_config = resolve_model(args.model)
    device = args.device or model_config["device"]
    model_name = model_config["full_name"]

    print(f"Running: {args.script}")
    print(f"Model:   {model_name}")
    print(f"Device:  {device}")

    module = importlib.import_module(args.script)
    asyncio.run(module.main(
        model=model_name,
        device=device,
        n_prompts=args.n_prompts,
    ))
