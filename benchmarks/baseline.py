""" Baseline benchmark. """

import asyncio
import httpx
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from utils.config import get_base_url, get_prompts
from utils.single_request import single_request


async def run_benchmark(client: httpx.AsyncClient, prompts: list[str], *, model: str, base_url: str) -> dict:
    """Run the benchmark and return the results."""
    n_requests = len(prompts)

    start = time.time()
    results = await asyncio.gather(
        *[single_request(client, prompt, model=model, base_url=base_url) for prompt in prompts],
        return_exceptions=True,
    )
    end = time.time()

    success_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    ttft_ms_percentiles = np.percentile([result["ttft_ms"] for result in success_results], [50, 90, 95, 99])
    tps_percentiles = np.percentile([result["tps"] for result in success_results], [50, 90, 95, 99])
    total_tokens = sum([result["total_tokens"] for result in success_results])
    throughput_total_tps = total_tokens / (end - start)

    return {
        "n_requests": n_requests,
        "failed_results": len(failed_results),
        "ttft_ms_percentiles": {
            "p50": float(ttft_ms_percentiles[0]),
            "p90": float(ttft_ms_percentiles[1]),
            "p95": float(ttft_ms_percentiles[2]),
            "p99": float(ttft_ms_percentiles[3]),
        },
        "tps_percentiles": {
            "p50": float(tps_percentiles[0]),
            "p90": float(tps_percentiles[1]),
            "p95": float(tps_percentiles[2]),
            "p99": float(tps_percentiles[3]),
        },
        "throughput_total_tps": throughput_total_tps,
    }


async def main(
    *,
    model: str,
    device: str,
    n_prompts: int | None = None,
):
    """Run the baseline benchmark."""
    base_url = get_base_url(device)
    prompts = get_prompts(n_prompts)
    async with httpx.AsyncClient(timeout=120.0) as client:
        results = await run_benchmark(client, prompts, model=model, base_url=base_url)

    output = {
        "experiment": "baseline",
        "device": device,
        "model": model,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }

    Path("results").mkdir(exist_ok=True)
    outfile = f"results/baseline_{device}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to {outfile}")
