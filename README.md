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
│   ├── prefix_cache_mps.json
│   ├── quantization_bf16_cuda.json
│   ├── quantization_int8_cuda.json
│   └── quantization_int4_cuda.json
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
| `qwen-0.5b` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | MPS |
| `qwen-1.5b-mps` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | MPS |
| `qwen-1.5b-cuda` | `Qwen/Qwen2.5-1.5B-Instruct` | CUDA |
| `qwen-1.5b-int8` | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | CUDA |
| `qwen-1.5b-int4` | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | CUDA |

## Metrics

All latency is reported in percentiles (P50/P90/P95/P99), never just mean. LLM
latency is right-skewed — mean is always higher than P50 due to outliers.

| Metric | What it measures | Phase |
| -------- | ----------------- | ------- |
| TTFT | Time to first token (ms) | Prefill |
| TPS | Tokens per second, per user | Decode |
| Throughput | Total tokens/sec across all concurrent requests | System |

## Quantization

Quantization reduces the precision of model weights to move fewer bytes per decode
step — directly attacking the memory-bandwidth bottleneck. But the method matters as
much as the precision level.

### CUDA — RTX 4090, Qwen2.5-1.5B-Instruct, 50 concurrent requests

| Precision | Model | TPS P50 | Throughput | Perplexity |
| --------- | ----- | ------- | ---------- | ---------- |
| BF16 | Qwen2.5-1.5B-Instruct | 212 | 1,634 TPS | 1.935 |
| INT8 (AWQ) | Qwen2.5-1.5B-Instruct-AWQ | 82 | 923 TPS | 1.986 |
| INT4 (GPTQ) | Qwen2.5-1.5B-Instruct-GPTQ-Int4 | 291 | 2,014 TPS | 1.949 |

### Reproducing

Each precision level requires a different vLLM server configuration:

```bash
# BF16 (baseline)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 --max-model-len 4096

# INT8 (AWQ)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --quantization awq --max-model-len 4096

# INT4 (GPTQ)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 \
  --quantization gptq --max-model-len 4096
```

Then run the benchmark against each:

```bash
uv run python -m main --script benchmarks.quantization --model qwen-1.5b-cuda --precision bf16
uv run python -m main --script benchmarks.quantization --model qwen-1.5b-int8 --precision int8
uv run python -m main --script benchmarks.quantization --model qwen-1.5b-int4 --precision int4
```

### Why INT8 is slower than full precision

The naive expectation — fewer bits = faster inference — breaks down here. AWQ
quantizes both weights *and* activations, which requires dequantization at runtime.
On a small 1.5B model, that dequantization overhead outweighs the memory bandwidth
savings from smaller weights.

GPTQ is weight-only quantization — it quantizes weights to INT4 but leaves activations
in full precision. Less overhead, and INT4 weights are half the size of INT8, so the
bandwidth savings are larger. Result: INT4 GPTQ is 37% faster than BF16, while INT8
AWQ is 61% *slower*.

This is a known tradeoff: AWQ's activation quantization overhead is amortized on larger
models (7B+) where memory savings dominate. On small models, weight-only methods like
GPTQ win.

### Quality is essentially free

Perplexity is nearly identical across all three precisions (1.935–1.986). At 1.5B
parameters, there's enough redundancy in the weights that precision loss barely
registers. The book confirms this — larger models are *less* sensitive to quantization
because each individual parameter matters less.

## Completed experiments

- **Concurrency sweep** — empirically derived optimal concurrency per device
- **Prefix caching** — shared system prompt KV cache reuse: 51% TTFT reduction on CUDA
- **Quantization** — BF16 vs INT8 (AWQ) vs INT4 (GPTQ): INT4 is 37% faster, INT8 is slower due to dequant overhead

## Planned experiments

- **Speculative decoding** — draft model validation: measure TPS improvement at varying batch sizes

## References

- Philip Kiely, [*Inference Engineering*](https://www.baseten.co/inference-engineering/) (2026)
- [vLLM documentation](https://docs.vllm.ai/)
- [MLX documentation](https://ml-explore.github.io/mlx/)
