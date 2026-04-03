# llm-serving-lab

## What this project is

A benchmarking and optimization playground for LLM inference, built to demonstrate
practical understanding of inference engineering across the full stack — from GPU
memory mechanics to production serving techniques.

The project systematically applies inference optimization techniques, measures the
effect of each, and documents the *why* grounded in theory. This makes it useful
both as a learning artifact and as a portfolio piece for inference engineering roles
and TA/RA applications at GaTech MSCS.

**The core thesis of this project (and of inference engineering generally):**
LLM decode is memory-bandwidth-bound, not compute-bound. The ops:byte ratio of
modern GPUs (~295 for H100 in FP16) far exceeds the arithmetic intensity of a
single-token decode step (~62 for a typical attention head). Every optimization
technique in this project is a direct attack on that bottleneck.

## Author background

- 6 years SWE experience; leads backend and infra
- Daily stack: FastAPI, Postgres, AWS
- Familiar with transformers (Karpathy GPT-2 follow-along: github.com/yashtotla/gpt-2)
- Comfortable with Python, venv/pip, Cursor
- New to inference engineering specifically

## Tech stack

- **Package manager**: `uv` (use `uv add` not `pip install`, `uv run` to execute scripts)
- **Python**: 3.11+
- **GPU environment (RunPod)**: RTX 4090 (24 GB VRAM), Ada Lovelace architecture
  - Supports: BF16, FP16, INT8 (W8A8), INT4 (GPTQ/AWQ)
  - Does NOT support FP8 natively (that requires Hopper / H100)
  - Inference engine: vLLM
- **Local environment (Mac M5, 16 GB unified memory)**: MLX via `mlx-lm`
- **Model**: Llama-3.2-1B-Instruct (small enough to fit everywhere, well-supported)
- **Serving wrapper**: FastAPI
- **Benchmarking**: custom Python scripts in `benchmarks/`
- **Analysis**: Jupyter notebooks in `notebooks/`

## Repo structure

```
llm-serving-lab/
├── benchmarks/
│   ├── baseline.py           # TTFT, TPS, latency percentiles at baseline
│   ├── prefix_cache.py       # Prefix caching speedup experiment
│   ├── quantization.py       # FP16 vs INT8 vs INT4 speed + quality
│   └── speculative.py        # Speculative decoding TPS comparison
├── configs/
│   ├── vllm_baseline.yaml    # vLLM server config: baseline FP16
│   ├── vllm_int8.yaml        # vLLM server config: INT8 quantized
│   └── vllm_speculative.yaml # vLLM server config: with draft model
├── results/                  # JSON/CSV output from benchmark runs, committed to git
├── notebooks/
│   └── analysis.ipynb        # Charts and analysis across experiments
├── serving/
│   └── main.py               # FastAPI wrapper around vLLM OpenAI-compatible API
├── mlx/
│   └── benchmark_mlx.py      # Same benchmark suite, run on Mac with mlx-lm
├── CLAUDE.md                 # This file
├── README.md
└── pyproject.toml
```

## Core concepts from the book (Philip Kiely, "Inference Engineering", 2026)

These are the theoretical foundations for each experiment. Reference them in code
comments and README explanations.

### The two phases of LLM inference

**Prefill**: Process the entire input sequence, compute attention for each token,
build the KV cache. This is **compute-bound** (high arithmetic intensity — many
matmuls on a large input matrix). Determines TTFT (time to first token).

**Decode**: Generate tokens one at a time, each requiring a full forward pass
loading all model weights. This is **memory-bandwidth-bound** (low arithmetic
intensity — vector-matrix multiply, loading weights for one token at a time).
Determines TPS (tokens per second).

### The ops:byte ratio and why decode is slow

Every GPU has an ops:byte ratio: compute (FLOPS) divided by memory bandwidth (GB/s).

- RTX 4090: ~330 TFLOPS FP16, 1 TB/s bandwidth → ops:byte ≈ 330
- H100: ~989 TFLOPS FP16, 3.35 TB/s → ops:byte ≈ 295

For inference to be balanced (neither resource idle), the algorithm's arithmetic
intensity must equal the hardware's ops:byte ratio.

Arithmetic intensity = FLOPs / bytes moved

For decode (single token, sequence length N=4096, head dim d=128):

- Memory: load Q/K/V matrices (each Nxd = ~4MB at FP16), load model weights
- Compute: vector-matrix multiply — far fewer ops per byte loaded
- Result: arithmetic intensity ≈ 62, far below GPU's ops:byte ratio of ~295-330
- **Conclusion**: Decode is memory-bound. Compute sits idle. This is the problem
  every technique in this project attacks.

### Key metrics

| Metric | Definition | Phase |
| --- | --- | --- |
| TTFT | Time to first token | Prefill |
| TPS (per-user) | Tokens per second received by a single user | Decode |
| ITL | Inter-token latency (1/TPS) | Decode |
| Throughput | Total tokens/sec across all users | System |
| P50/P90/P95/P99 | Latency percentiles (P99 matters for UX) | Both |

Always measure latency in percentiles, not just mean. LLM latency is right-skewed —
mean is always higher than P50 due to outliers. A P99 that is 5x P50 is bad UX.

### Technique 1: Quantization (Chapter 5.1)

**What it does**: Reduces numerical precision of model weights and activations
from FP16 to INT8 or INT4.

**Why it helps**:

- Decode (memory-bound): loads half as many bytes per weight → effectively doubles
  memory bandwidth → ~30-50% better TPS in practice (overhead prevents 2x)
- Prefill (compute-bound): lower-precision Tensor Cores have higher FLOPS

**Number formats relevant to RTX 4090 (Ada Lovelace)**:

- FP16: native training precision, baseline
- BF16: wider dynamic range than FP16, same size (supported on Ada+)
- INT8 (W8A8): weights and activations in 8-bit integer. Use GPTQ or AWQ.
- INT4 (GPTQ): aggressive quantization, noticeable quality drop on small models
- FP8: NOT natively supported on Ada Lovelace (requires Hopper / H100+)

**Quality sensitivity (least to most)**:
weights → activations → KV cache → attention layers

**How to measure quality impact**: perplexity score on a fixed test corpus.
Lower perplexity = better. After quantization, look for minimal perplexity increase.

**Quantization tools for vLLM**:

- `--quantization awq` or `--quantization gptq` for weight-only INT4
- `--quantization fp8` only on H100+ (skip on 4090)
- Pre-quantized models: `TheBloke/Llama-3.2-1B-Instruct-GPTQ` etc. on HuggingFace

### Technique 2: Prefix Caching / KV Cache Re-Use (Chapter 5.3)

**What it does**: Saves the KV cache from a request's shared prefix tokens and
re-uses it for subsequent requests that share that same prefix. Skips prefill on
cached tokens → reduces TTFT.

**Why it helps**: Building the KV cache during prefill is the expensive part of
TTFT. If two requests both start with a 500-token system prompt, only the first
one needs to compute that attention. All subsequent requests hit the cache.

**Critical implementation detail**: The prefix must match from the VERY FIRST token.
A single different token at position 0 breaks the entire cache, even if all later
tokens are identical.

**When it helps most**:

- Long shared system prompts (RAG scaffolds, agents, chatbots)
- Multi-turn conversations (each turn repeats all prior context)
- Code completion (same codebase context repeated)

**How to measure**: Compare TTFT on cache MISS (first request) vs cache HIT
(subsequent identical prefix). With a 500-token shared system prompt, expect
significant TTFT reduction on hits.

**vLLM config**: `--enable-prefix-caching` flag. Default in recent vLLM versions.

### Technique 3: Speculative Decoding (Chapter 5.2)

**What it does**: Uses a small "draft model" (or n-gram lookup) to speculatively
generate N draft tokens, then validates all N tokens in a single forward pass of
the target model. If accepted, generates N+1 tokens per forward pass instead of 1.

**Why it helps**: Decode is memory-bound with spare compute. Speculation uses that
idle compute to validate multiple draft tokens per pass, improving TPS.

**Only improves TPS, not TTFT.**

**Key factors affecting acceptance rate**:

- Temperature: higher temp → harder to predict → lower acceptance
- Domain match between draft and target model
- Sequence length of draft (more draft tokens → lower per-token acceptance)

**vLLM speculative decoding**: set `--speculative-model` to a smaller model
(e.g., `meta-llama/Llama-3.2-1B` as draft for a `3B` target) and
`--num-speculative-tokens 5`.

**Limitation**: Most effective at low batch sizes where compute is idle.
At high batch sizes, speculation overhead outweighs the benefit.

### Why MLX on Apple Silicon is interesting (Chapter 3.5)

Apple Silicon (M-series) uses unified memory — CPU and GPU share the same
physical memory pool. This eliminates the CPU↔GPU memory copy bottleneck
that CUDA-based systems have.

Implications:

- No OOM errors from loading model weights (uses system RAM directly)
- Memory bandwidth to GPU is lower than discrete GPU VRAM bandwidth
- At low batch sizes (single user), MLX can be competitive with discrete GPU
- At high concurrency, discrete GPU wins decisively on throughput

Comparing vLLM on 4090 vs MLX on M5 demonstrates these tradeoffs empirically.

## Three-day execution plan

### Day 1: Setup + Baseline

**Goal**: Working inference server + benchmarking harness measuring real numbers.

Tasks:

1. RunPod setup: Launch RTX 4090 pod with RunPod PyTorch 2.x template
2. Install vLLM: `pip install vllm` (inside pod, system Python is fine)
3. Pull model: `meta-llama/Llama-3.2-1B-Instruct`
4. Launch vLLM server:

   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-3.2-1B-Instruct \
     --dtype bfloat16 \
     --max-model-len 4096
   ```

5. Write `benchmarks/baseline.py`: send N requests, measure TTFT, TPS, P50/P90/P99
6. Write `serving/main.py`: thin FastAPI wrapper (plays to author's strength)
7. Save results to `results/baseline.json`
8. Add to README: explain the ops:byte ratio, why decode is memory-bound, what
   these numbers mean in context of hardware specs

**Mac (can do in parallel)**:

1. `uv add mlx-lm`
2. `mlx_lm.server --model mlx-community/Llama-3.2-1B-Instruct-4bit`
3. Run same `benchmarks/baseline.py` against local server, save to `results/baseline_mlx.json`

### Day 2: Optimizations

**Goal**: Measurable deltas from each technique, documented with the theory.

Tasks (RunPod):

1. **Quantization experiment** (`benchmarks/quantization.py`):
   - Launch vLLM with `--quantization awq` (use a pre-quantized AWQ model)
   - Run same benchmark suite
   - Measure: perplexity on 50-sentence test set, TTFT, TPS
   - Save: `results/quantization_int4.json`
   - Repeat with INT8 if available

2. **Prefix caching experiment** (`benchmarks/prefix_cache.py`):
   - Enable `--enable-prefix-caching` on vLLM
   - Design prompts with a shared 500-token system prompt + unique suffix
   - Send 10 identical-prefix requests, measure TTFT on request 1 (miss) vs 2-10 (hits)
   - Save: `results/prefix_cache.json`

3. **Speculative decoding** (`benchmarks/speculative.py`):
   - Launch vLLM with `--speculative-model` pointing to a tiny draft model
   - Measure TPS improvement vs baseline
   - Try at different batch sizes to show degradation at high concurrency
   - Save: `results/speculative.json`

### Day 3: Polish

**Goal**: Clean, reproducible, well-documented project that tells a story.

Tasks:

1. `notebooks/analysis.ipynb`: charts comparing all experiments
   - Bar chart: TTFT across techniques
   - Line chart: P50/P90/P99 latency distribution
   - Scatter: TPS vs batch size for baseline vs speculative
   - Table: quality (perplexity) vs speed tradeoff for quantization levels
2. Docker setup: `Dockerfile` + `docker-compose.yml` for reproducibility
3. README: full writeup with methodology, results, and the theory explaining each result
4. Final MLX vs GPU comparison table in README

## Benchmarking script contract

All benchmark scripts must output a JSON file with this schema:

```json
{
  "experiment": "baseline | prefix_cache | quantization_int8 | speculative",
  "environment": "runpod_4090 | mac_m5_mlx",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "timestamp": "ISO8601",
  "config": {},
  "results": {
    "n_requests": 100,
    "ttft_ms": { "p50": 0, "p90": 0, "p95": 0, "p99": 0, "mean": 0 },
    "tps_per_user": { "p50": 0, "p90": 0, "mean": 0 },
    "throughput_total_tps": 0,
    "perplexity": null
  }
}
```

Save all results to `results/` and commit them to git so experiments are reproducible.

## Style and conventions

- Comments in benchmark scripts should cite the relevant book section (e.g., `# Ch 5.3: prefix caching`)
- README explanations should explain the WHY, not just the WHAT
- Keep FastAPI serving code production-quality (proper error handling, async, typed)
- Use `httpx.AsyncClient` for concurrent benchmark requests
- Latency must always be reported in percentiles, never just mean
