""" Configuration for the project. """

import os

MODEL_REGISTRY = {
    # Same model, both environments — primary comparison pair
    "llama-1b-mps":   {"full_name": "mlx-community/Llama-3.2-1B-Instruct-4bit", "device": "mps"},
    "llama-1b-cuda":  {"full_name": "meta-llama/Llama-3.2-1B-Instruct",          "device": "cuda"},

    # Quantization experiment variants (CUDA only)
    "llama-1b-int8":  {"full_name": "meta-llama/Llama-3.2-1B-Instruct",          "device": "cuda"},  # vLLM --quantization awq
    "llama-1b-int4":  {"full_name": "TheBloke/Llama-3.2-1B-Instruct-GPTQ",       "device": "cuda"},

    # Tiny model just for local smoke testing
    "qwen-0.5b":      {"full_name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",  "device": "mps"},

    # Same model, both environments — secondary comparison pair
    "qwen-1.5b-cuda": {"full_name": "Qwen/Qwen2.5-1.5B-Instruct", "device": "cuda"},
    "qwen-1.5b-mps":  {"full_name": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "device": "mps"},
}

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
