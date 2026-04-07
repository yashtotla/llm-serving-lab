"""Semaphore-throttled wrapper around single_request."""

import asyncio
import httpx
from utils.single_request import single_request


async def throttled_request(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    prompt: str,
    *,
    model: str,
    base_url: str,
    system_prompt: str | None = None,
    additional_payload: dict | None = None,
) -> dict:
    """Run a single request, acquiring the semaphore first."""
    async with sem:
        return await single_request(
            client, prompt, model=model, base_url=base_url,
            system_prompt=system_prompt, additional_payload=additional_payload,
        )
