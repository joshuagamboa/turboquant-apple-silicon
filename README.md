# turboquant-apple-silicon

**TurboQuant KV Cache Quantization for llama.cpp on Apple Silicon (Metal/ARM)**

✅ **STATUS: FULLY FUNCTIONAL** — Production inference loop with Metal GPU acceleration, static linking stability, and full LLM observability (TTFT, TPS, token usage, latency).

A production-grade Rust integration that wraps the [TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant) of llama.cpp, enabling aggressive KV cache compression with minimal quality loss — optimized for Apple Silicon GPUs via Metal compute shaders.

---

## What Is TurboQuant?

TurboQuant is a **fork-only** set of Metal compute kernels for llama.cpp that quantises the KV cache at inference time. This dramatically reduces VRAM usage, allowing larger models and longer context windows to fit on Apple Silicon machines.

| Metric | FP16 KV (Baseline) | turbo3 | turbo2 |
|--------|---------------------|--------|--------|
| **Compression vs FP16** | 1.0× | ~4.6× | ~6.4× |
| **Memory Reduction** | — | ~78% | ~84% |
| **Decode Speed (M5 Max)** | Reference | ~100% | ~95% |

> **⚠️ TurboQuant is NOT in upstream llama.cpp.** You must use the [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **macOS** | 13.0+ (Ventura) | 15.0+ (Sequoia) |
| **Xcode** | 15.0+ | 16.2+ |
| **Rust** | 1.83+ (stable) | Latest stable |
| **CMake** | 3.28+ | Latest |
| **Architecture** | Apple Silicon (aarch64) | M3/M4/M5 |

---

## Project Structure

```
turboquant-apple-silicon/
├── Cargo.toml                  # Rust package manifest
├── build.rs                    # CMake + cc build orchestration
├── src/
│   ├── main.rs                 # CLI entry point + stats display
│   ├── ffi.rs                  # Hand-written C FFI bindings + LlamaTqEvalStats
│   ├── context.rs              # Safe Rust wrapper (TurboQuantCtx + InferenceStats)
│   └── shim/
│       └── llamatqshim.c       # C shim: inference loop + mach_absolute_time instrumentation
├── include/
│   └── llamatqshim.h           # C shim header (llamatq_eval_stats struct)
├── llama-cpp-turboquant/       # Fork (submodule or symlink)
├── scripts/
│   └── ci-smoke-test.sh        # CI smoke test
├── models/
│   └── .gitkeep                # GGUF models go here (gitignored)
└── README.md
```

---

## Quick Start

### 1. Clone the TurboQuant fork (into the project root)

```bash
cd turboquant-apple-silicon
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout 9c600bcd4   # Pinned stable commit
cd ..
```

> [!NOTE]
> The fork is gitignored. If you keep it elsewhere, set `LLAMA_TURBOQUANT_PATH` to its location.

### 2. Build

```bash
cargo build --release
```

### 3. Run

```bash
./target/release/turboquant-llama-rs models/your-model.gguf "Hello, world!" --temp 0.7 --top-p 0.9 --max-tokens 512
```

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `<model>` | (Required) | Path to the GGUF model file. |
| `[prompt]`| `"Hello, world!"` | The text prompt to start generation. |
| `--temp` | `0.0` | Temperature for sampling. `0.0` = greedy decoding. Higher values increase creativity. |
| `--top-p` | `1.0` | Nucleus sampling. `1.0` = disabled. Lower values (e.g. `0.9`) restrict to top cumulative tokens. |
| `--seed` | `0` | RNG seed for reproducible generation. |
| `--max-tokens` | `256` | Maximum tokens to generate before stopping. |
| `--ctx-size` | `8192` | Total context window size in tokens. Larger values use more KV cache memory. |
| `--batch-size` | `512` | Prompt-processing batch size in tokens. |
| `--verbose` | (Flag) | Print detailed diagnostics: memory breakdown, TTFT, prompt TPS, and per-phase timing. |

---

## Inference Statistics

After every successful run, TurboQuant prints a compact statistics block automatically:

```
─── Inference Stats ───────────────────────────────
  Latency:       1234.5 ms
  Tokens:        128 prompt → 256 generated  (384 total)
  Speed:         45.2 tok/s  (generation)
────────────────────────────────────────────────────
```

Pass `--verbose` for the full breakdown including **Time to First Token (TTFT)** and per-phase throughput:

```
─── Inference Stats (detailed) ─────────────────────
  Time to First Token:   89.3 ms
  Prompt Processing:     85.1 ms  (1504.7 tok/s, 128 tokens)
  Generation:            1149.4 ms  (45.2 tok/s, 256 tokens)
  Total Latency:         1234.5 ms
  Total Tokens:          384
─────────────────────────────────────────────────────
```

### Metrics Reference

| Metric | Where measured | Description |
|---|---|---|
| **TTFT** | C shim (`mach_absolute_time`) | Time from request start to end of first-token decode |
| **Latency** | C shim | Total wall-clock time (prompt processing + generation) |
| **Prompt TPS** | C shim | Prompt tokens processed per second |
| **Generation TPS** | C shim | Output tokens generated per second |
| **Token Usage** | C shim | Input prompt token count + output completion token count |

All timings use `mach_absolute_time()` — Apple Silicon's nanosecond-resolution monotonic clock — measured directly inside the C inference loop for the highest possible accuracy. TTFT in particular spans from the function entry to the completion of the first generated token's `llama_decode`, capturing the true user-perceived latency.

---

## Model Compatibility

| Model | Status | Notes |
|-------|--------|-------|
| TinyLlama 1.1B | ✅ Works | Great for CI/testing |
| Qwen 3.5 7B | ✅ Works | Good performance test |
| Qwen 3.5 35B-A3B | ✅ Works | Verified in fork benchmarks |
| Llama 3.1 8B | ✅ Works | Standard test model |
| Llama 3.1 70B | ⚠️ Works | Requires `-ngl` tuning, heavy VRAM |
| Mistral 7B | ✅ Works | Well-tested |
| Phi-3 Mini | ✅ Works | Lightweight option |

---

## Threading Safety

> **⚠️ TurboQuant + Metal contexts are NOT thread-safe.**

- Contexts must remain on a **single OS thread**
- Do **not** send contexts across threads or use in async executors without pinning
- The Rust wrapper enforces `!Send` + `!Sync` via `PhantomData<*mut ()>`

---

## Architecture

This crate uses a **C shim** (`llamatqshim.c`) to bridge Rust FFI with the llama.cpp C++ API. The shim is compiled by the `cc` crate, while the main llama.cpp fork is built via CMake — both orchestrated by `build.rs`.

```
Rust (main.rs)
  → InferenceStats display
  → Safe wrapper (context.rs / TurboQuantCtx)
    → FFI bindings (ffi.rs / LlamaTqEvalStats)
      → C shim (llamatqshim.c)
          ├─ mach_absolute_time() instrumentation (TTFT, TPS, latency)
          └─ llama.cpp TurboQuant fork (C++)
               └─ Metal compute kernels (GPU)
```

---

## Changelog

### [Unreleased] — LLM Inference Observability

Added standard LLM inference metrics collected entirely inside the C inference loop using `mach_absolute_time()` for nanosecond precision on Apple Silicon. Stats are returned to Rust via a new `llamatq_eval_stats` output struct and displayed after every run.

**New metrics:**
- **Time to First Token (TTFT)** — measures the latency from request start to completion of the first generated token's decode. Critical UX metric for chat applications.
- **Tokens Per Second (generation TPS)** — output throughput of the generation loop, excluding prompt processing.
- **Prompt Processing TPS** — throughput of the prompt ingestion phase (often GPU-bound and significantly faster than generation).
- **Token Usage (prompt / completion / total)** — input and output token counts for cost accounting and context management.
- **Total Latency** — full wall-clock time from call entry to final token.

**Implementation details:**
- `llamatq_eval_stats` struct added to `llamatqshim.h` and mirrored as `LlamaTqEvalStats` (`repr(C)`) in `ffi.rs`.
- Four `mach_absolute_time()` checkpoints injected into `llamatq_eval_with_sampling` in `llamatqshim.c`: at request start, after prompt decode, after first generated token's decode, and at generation loop exit.
- `out_stats` parameter is nullable — passing `NULL` (used by the `llamatq_eval` convenience wrapper) incurs zero overhead.
- `InferenceStats` Rust struct added to `context.rs` with `print_compact()`, `print_verbose()`, and `Display` implementations.
- `eval_with_sampling()` return type changed from `Result<i32, ...>` to `Result<InferenceStats, ...>`.
- Error codes from the C shim (`-1`, `-2`, `-3`) now have human-readable descriptions.
- C shim API version bumped `1 → 2` (reflected in both `build.rs` define and `EXPECTED_API_VERSION` in `context.rs`).
- Two output modes: compact (always shown) and detailed/verbose (with `--verbose` flag).

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
