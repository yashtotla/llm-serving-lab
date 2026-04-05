"""Shared streaming request helper used by all benchmark scripts."""

import json
import time
import httpx


def _parse_sse_data_line(line: str) -> str | None:
    """Return the payload after 'data:' for an SSE line, or None if not a data line."""
    if not line.startswith("data:"):
        return None
    return line[5:].lstrip()


async def single_request(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    model: str,
    base_url: str,
    system_prompt: str | None = None,
) -> dict:
    """Stream chat completions; TTFT is ms until first chunk with non-empty delta content."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_time = time.time()
    first_content_time: float | None = None
    full_content_parts: list[str] = []
    completion_tokens: int | None = None

    async with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
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
