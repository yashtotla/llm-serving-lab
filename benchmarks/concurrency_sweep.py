"""Concurrency sweep benchmark.

Runs the same prompt set at increasing concurrency levels to find the point
where throughput plateaus and TTFT starts degrading — i.e. optimal concurrency.

Concurrency is controlled via an asyncio.Semaphore: all prompts are launched
concurrently, but at most `concurrency` requests are in-flight at once.
"""

import asyncio
import httpx
import json
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from config import get_base_url, get_prompts
from utils.single_request import single_request

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]


async def _throttled_request(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    prompt: str,
    *,
    model: str,
    base_url: str,
) -> dict:
    """Run a single request, acquiring the semaphore first."""
    async with sem:
        return await single_request(client, prompt, model=model, base_url=base_url)


async def run_at_concurrency(
    client: httpx.AsyncClient,
    prompts: list[str],
    *,
    model: str,
    base_url: str,
    concurrency: int,
) -> dict:
    """Fire all prompts with at most `concurrency` in-flight at once."""
    sem = asyncio.Semaphore(concurrency)

    start = time.time()
    results = await asyncio.gather(
        *[
            _throttled_request(sem, client, prompt, model=model, base_url=base_url)
            for prompt in prompts
        ],
        return_exceptions=True,
    )
    end = time.time()

    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    if not successes:
        return {
            "concurrency": concurrency,
            "n_requests": len(prompts),
            "failed": len(failures),
            "ttft_ms": None,
            "tps_per_user": None,
            "throughput_total_tps": 0,
        }

    ttft_vals = [r["ttft_ms"] for r in successes]
    tps_vals = [r["tps"] for r in successes]
    total_tokens = sum(r["total_tokens"] for r in successes)

    ttft_pcts = np.percentile(ttft_vals, [50, 90, 95, 99])
    tps_pcts = np.percentile(tps_vals, [50, 90, 95, 99])

    return {
        "concurrency": concurrency,
        "n_requests": len(prompts),
        "failed": len(failures),
        "ttft_ms": {
            "p50": float(ttft_pcts[0]),
            "p90": float(ttft_pcts[1]),
            "p95": float(ttft_pcts[2]),
            "p99": float(ttft_pcts[3]),
            "mean": float(np.mean(ttft_vals)),
        },
        "tps_per_user": {
            "p50": float(tps_pcts[0]),
            "p90": float(tps_pcts[1]),
            "p95": float(tps_pcts[2]),
            "p99": float(tps_pcts[3]),
            "mean": float(np.mean(tps_vals)),
        },
        "throughput_total_tps": total_tokens / (end - start),
    }


async def main(
    *,
    model: str,
    device: str,
    n_prompts: int | None = None,
    concurrency_levels: list[int] = CONCURRENCY_LEVELS,
):
    """Sweep across concurrency levels and save combined results."""
    base_url = get_base_url(device)
    prompts = get_prompts(n_prompts)

    sweep_results: list[dict] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for level in concurrency_levels:
            print(f"Running concurrency={level} …")
            result = await run_at_concurrency(
                client, prompts, model=model, base_url=base_url, concurrency=level,
            )
            sweep_results.append(result)

            # Quick summary line
            if result["ttft_ms"]:
                print(
                    f"  TTFT p50={result['ttft_ms']['p50']:.0f}ms  "
                    f"throughput={result['throughput_total_tps']:.1f} tok/s  "
                    f"failed={result['failed']}"
                )

    output = {
        "experiment": "concurrency_sweep",
        "device": device,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "concurrency_levels": concurrency_levels,
        "results": sweep_results,
    }

    Path("results").mkdir(exist_ok=True)
    outfile = f"results/concurrency_sweep_{device}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to {outfile}")
