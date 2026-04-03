""" Baseline benchmark. """

import asyncio
import httpx
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path


PROMPTS = [
    # General knowledge
    "Explain how a CPU executes an instruction from fetch to retire.",
    "What causes ocean tides and how do they differ from waves?",
    "Describe the process of photosynthesis in simple terms.",
    "How does a transistor work at the physical level?",
    "What is the difference between TCP and UDP?",
    # Reasoning / analysis
    "Compare and contrast microservices and monolithic architectures.",
    "Why is the speed of light considered a universal speed limit?",
    "What are the trade-offs between consistency and availability in distributed systems?",
    "Explain why sorting algorithms have a theoretical lower bound of O(n log n).",
    "What makes hash tables O(1) on average but O(n) in the worst case?",
    # Creative / open-ended
    "Write a short poem about debugging code at 3 AM.",
    "Describe a futuristic city powered entirely by renewable energy.",
    "Invent a new board game and explain its rules.",
    "Write a dialogue between a compiler and an interpreter arguing about which is better.",
    "Create a short fable about a server that never went down.",
    # Programming
    "Write a Python function to find the longest palindromic substring.",
    "Explain how garbage collection works in Java.",
    "What is the difference between a mutex and a semaphore?",
    "Describe how a B-tree index speeds up database queries.",
    "Write a recursive solution for the Tower of Hanoi problem in Python.",
    # Math and science
    "Explain the intuition behind Bayes' theorem with an example.",
    "What is eigenvalue decomposition and why does it matter in machine learning?",
    "Describe the double-slit experiment and what it reveals about quantum mechanics.",
    "How does gradient descent find the minimum of a loss function?",
    "Explain the difference between correlation and causation with examples.",
    # History and society
    "What caused the fall of the Roman Empire?",
    "Explain the significance of the Gutenberg printing press.",
    "How did the Industrial Revolution change labor and society?",
    "What were the main causes of World War I?",
    "Describe the impact of the internet on modern democracy.",
    # Systems and infrastructure
    "How does DNS resolution work from browser to IP address?",
    "Explain how TLS establishes a secure connection.",
    "What happens when you type a URL into a browser and press Enter?",
    "Describe how a load balancer distributes traffic across servers.",
    "What is the CAP theorem and why does it matter for databases?",
    # AI and ML
    "Explain the transformer architecture in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "How does backpropagation compute gradients in a neural network?",
    "What is attention in neural networks and why was it a breakthrough?",
    "Explain the bias-variance tradeoff in machine learning.",
    # Practical tasks
    "Write a SQL query to find the second highest salary in a table.",
    "Explain how to set up a CI/CD pipeline for a Python project.",
    "What are the best practices for designing a REST API?",
    "Describe how to implement rate limiting in a web application.",
    "Write a bash one-liner to find the 10 largest files in a directory.",
    # Miscellaneous
    "What is the Fermi paradox and what are the leading explanations?",
    "Explain how CRISPR gene editing works.",
    "What are the key differences between IPv4 and IPv6?",
    "Describe how a blockchain achieves consensus without a central authority.",
    "Explain the concept of entropy in both thermodynamics and information theory.",
]

BASE_URL = "http://localhost:8000"

def _parse_sse_data_line(line: str) -> str | None:
    """Return the payload after 'data:' for an SSE line, or None if not a data line."""
    if not line.startswith("data:"):
        return None
    return line[5:].lstrip()

async def single_request(client: httpx.AsyncClient, prompt: str, *, model: str) -> dict:
    """Stream chat completions; TTFT is ms until first chunk with non-empty delta content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_time = time.time()
    first_content_time: float | None = None
    full_content_parts: list[str] = []
    completion_tokens: int | None = None

    async with client.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            raw = _parse_sse_data_line(line)
            if raw is None:
                continue
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue
                now = time.time()
                if first_content_time is None:
                    first_content_time = now
                full_content_parts.append(content)

            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

    end_time = time.time()

    if first_content_time is None:
        raise RuntimeError("Stream ended without any chunk containing non-empty delta content")

    # OpenAI/vLLM streaming: one generated token per non-empty delta.content (not spec-guaranteed).
    total_tokens = (
        completion_tokens if completion_tokens is not None else len(full_content_parts)
    )

    decode_s = end_time - first_content_time
    tps = (total_tokens / decode_s) if decode_s > 0 else float("inf")
    ttft_ms = (first_content_time - start_time) * 1000

    return {"ttft_ms": ttft_ms, "tps": tps, "total_tokens": total_tokens}


async def run_benchmark(client: httpx.AsyncClient, prompts: list[str], *, model: str) -> dict:
    """Run the benchmark and return the results."""
    n_requests = len(prompts)

    start = time.time()
    results = await asyncio.gather(
        *[single_request(client, prompt, model=model) for prompt in prompts],
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
    prompts = PROMPTS[:n_prompts] if n_prompts else PROMPTS
    async with httpx.AsyncClient(timeout=120.0) as client:
        results = await run_benchmark(client, prompts, model=model)

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
