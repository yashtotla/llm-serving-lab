"""Benchmark quantization. Measures perplexity + latency."""

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from datasets import load_dataset

from utils.config import get_base_url, get_max_concurrency
from utils.throttled_request import throttled_request


def get_wikitext_passages(n: int = 50) -> list[str]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    passages = [row["text"] for row in dataset if len(row["text"]) > 200]
    return passages[:n]


async def measure_perplexity(
    client: httpx.AsyncClient,
    passages: list[str],
    *,
    model: str,
    base_url: str,
    max_concurrency: int,
) -> dict:
    """Measure the perplexity of a model on a list of passages."""
    n_prompts = len(passages)
    sem = asyncio.Semaphore(max_concurrency)
    additional_payload = {
        "logprobs": True,
        "top_logprobs": 1,
        "temperature": 0,
    }

    start = time.time()
    results = await asyncio.gather(
        *[
            throttled_request(
                sem, client, passage,
                model=model, base_url=base_url,
                additional_payload=additional_payload,
            )
            for passage in passages
        ],
        return_exceptions=True,
    )
    end = time.time()

    success_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    if not success_results:
        return {
            "n_prompts": n_prompts,
            "failed_results": len(failed_results),
            "perplexity": None,
        }

    # Perplexity is a model property, not a per-request latency metric,
    # so percentiles don't apply — a single mean is the right summary.
    perplexities = []
    for result in success_results:
        token_logprobs = [t["logprob"] for t in result.get("logprobs", [])]
        if token_logprobs:
            perplexities.append(math.exp(-sum(token_logprobs) / len(token_logprobs)))

    perplexity = sum(perplexities) / len(perplexities) if perplexities else None

    ttft_ms_percentiles = np.percentile([r["ttft_ms"] for r in success_results], [50, 90, 95, 99])
    tps_percentiles = np.percentile([r["tps"] for r in success_results], [50, 90, 95, 99])
    total_tokens = sum(r["total_tokens"] for r in success_results)
    throughput_total_tps = total_tokens / (end - start)

    return {
        "n_prompts": n_prompts,
        "failed_results": len(failed_results),
        "perplexity": perplexity,
        "ttft_ms": {
            "p50": float(ttft_ms_percentiles[0]),
            "p90": float(ttft_ms_percentiles[1]),
            "p95": float(ttft_ms_percentiles[2]),
            "p99": float(ttft_ms_percentiles[3]),
        },
        "tps": {
            "p50": float(tps_percentiles[0]),
            "p90": float(tps_percentiles[1]),
            "p95": float(tps_percentiles[2]),
            "p99": float(tps_percentiles[3]),
        },
        "throughput_total_tps": throughput_total_tps,
    }


async def main(model: str, device: str, precision: str, n_prompts: int = 50):
    """Run the quantization benchmark.

    Args:
        model: Full model name (e.g. meta-llama/Llama-3.2-1B-Instruct).
        device: Target device (cuda or mps).
        precision: Precision label for the output file (bf16, int8, int4).
        n_prompts: Number of WikiText passages to evaluate.
    """
    base_url = get_base_url(device)
    max_concurrency = get_max_concurrency(device)
    passages = get_wikitext_passages(n_prompts)

    async with httpx.AsyncClient(timeout=120.0) as client:
        results = await measure_perplexity(
            client, passages,
            model=model, base_url=base_url, max_concurrency=max_concurrency,
        )

    output = {
        "experiment": f"quantization_{precision}",
        "device": device,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"precision": precision, "n_prompts": n_prompts},
        "results": results,
    }

    Path("results").mkdir(exist_ok=True)
    outfile = f"results/quantization_{precision}_{device}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to {outfile}")
