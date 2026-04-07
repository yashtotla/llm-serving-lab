""" Configuration for the project. """

import os
from dotenv import load_dotenv
from .constants import MODEL_REGISTRY, PROMPTS, SYSTEM_PROMPT

load_dotenv()

def resolve_model(alias: str) -> dict:
    if alias in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias]
    # fall through — treat as a full HuggingFace model name
    return {"full_name": alias, "device": None}


def get_base_url(device: str) -> str:
    """Return the inference server URL based on the target device.

    - cuda: uses RUNPOD_URL from .env (vLLM on RunPod)
    - mps:  uses http://localhost:{PORT} (mlx-lm local server)
    """
    if device == "cuda":
        url = os.environ.get("RUNPOD_URL")
        if not url:
            raise RuntimeError("RUNPOD_URL environment variable is not set")
        return url.rstrip("/")
    port = os.environ.get("PORT", "8080")
    return f"http://localhost:{port}"


def get_prompts(n_prompts: int | None = None) -> list[str]:
    """Return the prompts based on the number of prompts."""
    return PROMPTS[:n_prompts] if n_prompts else PROMPTS


def get_system_prompt() -> str:
    """Return the system prompt."""
    return SYSTEM_PROMPT


def get_max_concurrency(device: str) -> int:
    """Return the max concurrency based on the device."""
    if device == "cuda":
        return 16
    return 4
