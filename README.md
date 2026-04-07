# llm-serving-lab

LLM decode is **memory-bandwidth-bound**, not compute-bound. The ops:byte ratio of a
modern GPU (e.g. ~330 for RTX 4090 in FP16) far exceeds the arithmetic intensity of
a single-token decode step (~62 for a typical attention head). Every optimization in
this project is a direct attack on that bottleneck.

| Phase | Bottleneck | Key metric |
| ------- | ----------- | ------------ |
| **Prefill** — process the full input, build KV cache | Compute-bound (large matmuls) | TTFT (time to first token) |
| **Decode** — generate tokens one at a time, each loading all weights | Memory-bandwidth-bound | TPS (tokens per second) |

This project measures baseline inference performance across GPU (vLLM on RunPod RTX
4090) and Apple Silicon (MLX on Mac M5), then systematically applies optimization
techniques — quantization, prefix caching, speculative decoding — and documents the
measured impact of each.

## Baseline results

50 concurrent requests, Qwen2.5-1.5B-Instruct, streaming via OpenAI-compatible API.

> Note: baseline was run before the concurrency sweep, using full concurrency (50).
> Results reflect maximum throughput, not optimized latency.

### CUDA — RTX 4090 (vLLM, BF16)

| Metric | P50 | P90 | P95 | P99 |
| -------- | ----- | ----- | ----- | ----- |
| TTFT (ms) | 7,708 | 7,738 | 7,739 | 7,751 |
| TPS (per-user) | 183.7 | 187.3 | 190.3 | 194.6 |

**Throughput**: 1,506 tokens/sec across all requests.

### MPS — Mac M5 (MLX, 4-bit quantized)

| Metric | P50 | P90 | P95 | P99 |
| -------- | ----- | ----- | ----- | ----- |
| TTFT (ms) | 2,924 | 16,130 | 16,131 | 18,722 |
| TPS (per-user) | 34.7 | 40.3 | 40.5 | 42.1 |

**Throughput**: 704 tokens/sec across all requests.

### What the numbers tell us

- **TPS**: The 4090 delivers ~5x the per-user decode throughput of the M5. This is
  expected — discrete VRAM bandwidth (1 TB/s on 4090) dwarfs unified memory bandwidth
  on Apple Silicon.
- **TTFT**: The M5 has *lower* P50 TTFT (2.9s vs 7.7s), but this is misleading —
  CUDA TTFT includes ~2s round-trip network latency (Mac → RunPod proxy). True
  on-GPU TTFT was not measured in this experiment due to RunPod proxy overhead.
  M5's P99 TTFT explodes to 18.7s — requests that arrive late queue behind earlier
  decodes in the single-threaded MLX server.
- **Throughput**: System-wide, the 4090 pushes 2x the total tokens/sec even though
  the M5 model is already 4-bit quantized (smaller weights, fewer bytes to move).

## Concurrency sweep

Firing all requests at once saturates the server. The concurrency sweep empirically
finds the sweet spot — the concurrency level where throughput plateaus but TTFT
hasn't yet exploded.

| Device | Sweet spot | TTFT P50 | Throughput |
| -------- | ---------- | -------- | ---------- |
| MPS (1.5B) | 4 | 341ms | 292 TPS |
| CUDA (1.5B) | 16 | 620ms | 1,835 TPS |

Beyond these points, each additional concurrent request competes for the same memory
bandwidth during decode — per-user TPS drops faster than total throughput rises.

## Prefix caching

Prefix caching saves the KV cache from a shared system prompt and re-uses it for
subsequent requests, skipping redundant prefill work. This directly reduces TTFT.

### CUDA — RTX 4090 (vLLM, BF16)

| Metric | Value |
| -------- | ----- |
| Cache miss TTFT | 2,690ms |
| Cache hit TTFT P50 | 1,318ms (51% reduction) |
| Cache hit TTFT P90 | 8,997ms |
| Cache hit TPS P50 | 190.2 |
| Throughput | 1,088 TPS |

vLLM confirmed prefix cache hit rate: 95%. P90+ spikes are attributed to network
latency variance through the RunPod proxy, not vLLM queuing — server logs confirmed
`Waiting: 0 reqs` throughout the run.

### MPS — Mac M5 (MLX, 4-bit quantized)

| Metric | Value |
| -------- | ----- |
| Cache miss TTFT | 753ms |
| Cache hit TTFT P50 | 520ms (31% reduction) |
| Cache hit TTFT P90 | 675ms |
| Cache hit TPS P50 | 44.8 |
| Throughput | 168 TPS |

## Project structure

```md
llm-serving-lab/
├── main.py                    # CLI entry point — routes to benchmark scripts
├── utils/
│   ├── config.py              # Model registry + environment-aware URL resolution
│   ├── constants.py           # Shared prompts + generation parameters
│   ├── single_request.py      # Single streaming request helper
│   └── throttled_request.py   # Rate-limited concurrent request dispatcher
├── benchmarks/
│   ├── baseline.py            # TTFT, TPS, latency percentiles at baseline
│   ├── concurrency_sweep.py   # Find optimal concurrency per device
│   └── prefix_cache.py        # Prefix caching speedup experiment
├── results/                   # JSON output from benchmark runs
│   ├── baseline_cuda.json
│   ├── baseline_mps.json
│   ├── concurrency_sweep_cuda.json
│   ├── concurrency_sweep_mps.json
│   ├── prefix_cache_cuda.json
│   └── prefix_cache_mps.json
├── .env                       # RUNPOD_URL, PORT, HF_TOKEN — gitignored
└── pyproject.toml
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

```bash
uv sync
```

### Environment variables

Create a `.env` file at the project root:

```env
HF_TOKEN=your_huggingface_token
RUNPOD_URL=https://your-runpod-proxy-url
PORT=8080
```

- `RUNPOD_URL` — used when running benchmarks against a CUDA device (vLLM on RunPod)
- `PORT` — local server port for MLX benchmarks (defaults to `8080`)

## Usage

### Running the local MLX server

```bash
uv run mlx_lm.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8080
```

### Running benchmarks

All benchmarks are invoked through `main.py` using model aliases defined in `config.py`:

```bash
# Mac (MLX) — full benchmark
uv run python -m main --script benchmarks.baseline --model qwen-1.5b-mps

# RunPod (CUDA) — full benchmark
uv run python -m main --script benchmarks.baseline --model qwen-1.5b-cuda

# Quick smoke test — 5 prompts only
uv run python -m main --script benchmarks.baseline --model qwen-0.5b --n-prompts 5
```

### Model aliases

| Alias | Model | Device |
| ------- | ------- | -------- |
| `llama-1b-mps` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | MPS |
| `llama-1b-cuda` | `meta-llama/Llama-3.2-1B-Instruct` | CUDA |
| `llama-1b-int8` | `meta-llama/Llama-3.2-1B-Instruct` | CUDA |
| `llama-1b-int4` | `TheBloke/Llama-3.2-1B-Instruct-GPTQ` | CUDA |
| `qwen-0.5b` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | MPS |
| `qwen-1.5b-mps` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | MPS |
| `qwen-1.5b-cuda` | `Qwen/Qwen2.5-1.5B-Instruct` | CUDA |

## Metrics

All latency is reported in percentiles (P50/P90/P95/P99), never just mean. LLM
latency is right-skewed — mean is always higher than P50 due to outliers.

| Metric | What it measures | Phase |
| -------- | ----------------- | ------- |
| TTFT | Time to first token (ms) | Prefill |
| TPS | Tokens per second, per user | Decode |
| Throughput | Total tokens/sec across all concurrent requests | System |

## Completed experiments

- **Concurrency sweep** — empirically derived optimal concurrency per device
- **Prefix caching** — shared system prompt KV cache reuse: 51% TTFT reduction on CUDA

## Planned experiments

- **Quantization** — FP16 vs INT8 vs INT4: measure TPS gain and perplexity cost
- **Speculative decoding** — draft model validation: measure TPS improvement at varying batch sizes

## References

- Philip Kiely, [*Inference Engineering*](https://www.baseten.co/inference-engineering/) (2026)
- [vLLM documentation](https://docs.vllm.ai/)
- [MLX documentation](https://ml-explore.github.io/mlx/)
