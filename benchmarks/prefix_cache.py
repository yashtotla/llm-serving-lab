"""Benchmark prefix cache."""

import asyncio
import httpx
import json
import time
import numpy as np
from datetime import datetime
from config import get_base_url, get_prompts, get_system_prompt
from utils.single_request import single_request
from pathlib import Path

async def run_benchmark(client: httpx.AsyncClient, prompts: list[str], *, model: str, base_url: str, system_prompt: str) -> dict:
    """Run the benchmark and return the results."""
    n_requests = len(prompts)

    miss_result = await single_request(client, prompts[0], model=model, base_url=base_url, system_prompt=system_prompt)
    if isinstance(miss_result, Exception):
        raise RuntimeError(f"Cache miss request failed: {miss_result.error}")

    hit_start = time.time()
    hit_results = await asyncio.gather(
        *[single_request(client, prompt, model=model, base_url=base_url, system_prompt=system_prompt) for prompt in prompts[1:]],
        return_exceptions=True,
    )
    hit_end = time.time()

    hit_failed_results = [r for r in hit_results if isinstance(r, Exception)]
    hit_success_results = [r for r in hit_results if not isinstance(r, Exception)]

    hit_ttft_ms_percentiles = np.percentile([result["ttft_ms"] for result in hit_success_results], [50, 90, 95, 99])
    hit_tps_percentiles = np.percentile([result["tps"] for result in hit_success_results], [50, 90, 95, 99])
    hit_total_tokens = sum([result["total_tokens"] for result in hit_success_results])
    hit_throughput_total_tps = hit_total_tokens / (hit_end - hit_start)

    return {
        "n_requests": n_requests,
        "cache_miss": {
            "ttft_ms": miss_result["ttft_ms"],
            "tps": miss_result["tps"],
            "total_tokens": miss_result["total_tokens"],
        },
        "cache_hit": {
            "failed_results": len(hit_failed_results),
            "ttft_ms_percentiles": {
                "p50": float(hit_ttft_ms_percentiles[0]),
                "p90": float(hit_ttft_ms_percentiles[1]),
                "p95": float(hit_ttft_ms_percentiles[2]),
                "p99": float(hit_ttft_ms_percentiles[3]),
            },
            "tps_percentiles": {
                "p50": float(hit_tps_percentiles[0]),
                "p90": float(hit_tps_percentiles[1]),
                "p95": float(hit_tps_percentiles[2]),
                "p99": float(hit_tps_percentiles[3]),
            },
            "throughput_total_tps": hit_throughput_total_tps,
        },
    }

async def main(model: str, device: str, n_prompts: int | None = None):
    """Run the prefix cache benchmark."""
    base_url = get_base_url(device)
    prompts = get_prompts(n_prompts)
    system_prompt = get_system_prompt()

    async with httpx.AsyncClient(timeout=120.0) as client:
        results = await run_benchmark(client, prompts, model=model, base_url=base_url, system_prompt=system_prompt)

    output = {
        "experiment": "prefix_cache",
        "device": device,
        "model": model,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }

    Path("results").mkdir(exist_ok=True)
    outfile = f"results/prefix_cache_{device}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to {outfile}")
