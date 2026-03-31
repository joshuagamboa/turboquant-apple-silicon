# turboquant-apple-silicon

**TurboQuant KV Cache Quantization for llama.cpp on Apple Silicon (Metal/ARM)**

✅ **STATUS: FULLY FUNCTIONAL** — Production inference loop with Metal GPU acceleration, multi-turn interactive chat TUI, stateful context-window management, chat template auto-detection, and full LLM observability (TTFT, TPS, token usage, latency).

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
├── Cargo.toml                  # Rust package manifest (incl. tuinix, clap)
├── build.rs                    # CMake + cc build orchestration (API version gate)
├── src/
│   ├── main.rs                 # CLI entry point — one-shot and --chat mode
│   ├── ffi.rs                  # Hand-written C FFI bindings (API v3)
│   ├── context.rs              # Safe Rust wrapper (TurboQuantCtx, InferenceStats)
│   ├── chat.rs                 # ChatSession — multi-turn state + KV windowing
│   ├── template.rs             # Chat template engine (ChatML / Llama-3 / Mistral)
│   ├── tui.rs                  # Interactive TUI (tuinix frame-based rendering)
│   └── shim/
│       └── llamatqshim.c       # C shim: inference, KV ops, streaming callback
├── include/
│   └── llamatqshim.h           # C shim header (API v3: 12 functions)
├── llama-cpp-turboquant/       # Fork (submodule or local clone)
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

### 3a. One-shot inference

```bash
./target/release/turboquant-llama-rs models/your-model.gguf "Hello!" \
  --temp 0.7 --top-p 0.9 --max-tokens 512
```

### 3b. Interactive multi-turn chat (TUI)

```bash
./target/release/turboquant-llama-rs models/your-model.gguf --chat
```

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `<model>` | (Required) | Path to the GGUF model file. |
| `[prompt]` | `"Hello, world!"` | Prompt for one-shot mode (ignored in `--chat`). |
| `--chat` | (Flag) | Launch the interactive multi-turn chat TUI. |
| `--template` | (Auto) | Override chat template: `chatml`, `llama3`, or `mistral`. |
| `--temp` | `0.0` | Temperature for sampling. `0.0` = greedy. Higher = more creative. |
| `--top-p` | `1.0` | Nucleus sampling. `1.0` = disabled. |
| `--seed` | `0` | RNG seed for reproducible generation. |
| `--max-tokens` | `512` | Maximum tokens to generate per turn. |
| `--ctx-size` | `8192` | Total context window size in tokens. |
| `--batch-size` | `512` | Prompt-processing batch size in tokens. |
| `--verbose` | (Flag) | Print detailed diagnostics: memory breakdown, TTFT, per-phase timing. |

---

## Interactive Chat TUI

Launch with `--chat` to enter a frame-based terminal UI:

```
 TurboQuant Chat │ Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf │ ChatML │ 312/8192 tokens
─────────────────────────────────────────────────────────────────────────────────
  You › Who wrote Don Quixote?

   AI › Miguel de Cervantes wrote Don Quixote, published in two parts in 1605
        and 1615. It is widely regarded as the first modern novel.

─────────────────────────────────────────────────────────────────────────────────
 ❯ _
─────────────────────────────────────────────────────────────────────────────────
 45.2 tok/s  TTFT 83ms │ Ctrl+C: quit  /reset  /help  /stats  PgUp/Dn: scroll
```

### TUI Keyboard Controls

| Key | Action |
|-----|--------|
| `Enter` | Submit message / run inference |
| `Ctrl+C` | Exit chat |
| `←` / `→` | Move cursor in input |
| `Home` / `End` | Jump to start / end of input |
| `Backspace` / `Del` | Delete character |
| `PgUp` / `PgDn` | Scroll message history |

### TUI Slash Commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear message history and reset KV cache |
| `/help` | Show available commands |
| `/stats` | Show last inference statistics (TTFT, TPS, tokens) |

### Streaming

Tokens are streamed live into the TUI as they are generated — each token triggers a differential frame update via the tuinix callback mechanism, giving you real-time output without terminal corruption.

---

## Context Window Management

Long conversations eventually exceed the model's context window. TurboQuant implements a two-strategy windowing system to handle this gracefully:

### Strategy 1 — KV Shift (Primary)
Uses `llama_memory_seq_rm` + `llama_memory_seq_add` to **evict old tokens in-place** from the KV cache without clearing the rest of the context. This preserves recent conversation turns and is extremely fast (no re-encoding).

### Strategy 2 — Re-process (Fallback)
If the KV shift cannot reduce usage sufficiently, the session clears the full KV cache and re-encodes the most recent portion of the conversation history. Slower but guarantees a clean state.

The transition is invisible to the user — the chat continues without interruption.

---

## Chat Template Auto-Detection

The template engine reads GGUF metadata keys (`tokenizer.chat_template`, `general.architecture`) to automatically identify and apply the correct chat format:

| Template | Detected From | Format |
|----------|--------------|--------|
| **ChatML** | `tokenizer.chat_template` contains `im_start`, or Qwen architecture | `<|im_start|>system…<|im_end|>` |
| **Llama-3** | Architecture `llama` or template mentions `<|start_header_id|>` | `<|start_header_id|>user<|end_header_id|>…` |
| **Mistral Instruct** | Architecture `mistral` or template mentions `[INST]` | `[INST] … [/INST]` |

Override with `--template chatml|llama3|mistral` if auto-detection is wrong.

**Default system prompt** (used when none is set by the model):
> *"You are a helpful, accurate, concise AI assistant. Follow the user's instructions, ask clarifying questions when needed, and avoid making up facts."*

---

## Inference Statistics

After every one-shot run, TurboQuant prints a compact statistics block:

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

In `--chat` mode, live stats (TPS + TTFT) are shown in the TUI status bar after each response. Use `/stats` to view them at any time.

### Metrics Reference

| Metric | Where measured | Description |
|---|---|---|
| **TTFT** | C shim (`mach_absolute_time`) | Time from request start to end of first-token decode |
| **Latency** | C shim | Total wall-clock time (prompt processing + generation) |
| **Prompt TPS** | C shim | Prompt tokens processed per second |
| **Generation TPS** | C shim | Output tokens generated per second |
| **Token Usage** | C shim | Input prompt token count + output completion token count |
| **Context Used** | KV cache query | Current KV cache occupancy (shown in TUI header) |

All timings use `mach_absolute_time()` — Apple Silicon's nanosecond-resolution monotonic clock — measured directly inside the C inference loop.

---

## Model Compatibility

| Model | Status | Chat Template | Notes |
|-------|--------|--------------|-------|
| TinyLlama 1.1B | ✅ Works | ChatML | Great for CI/testing |
| Qwen 2.5 0.5B / 7B | ✅ Works | ChatML (auto) | Fast, excellent for chat |
| Qwen 3.5 35B-A3B | ✅ Works | ChatML (auto) | Verified in fork benchmarks |
| Llama 3.2 1B / 3B | ✅ Works | Llama-3 (auto) | Lightweight instruct models |
| Llama 3.1 8B | ✅ Works | Llama-3 (auto) | Standard test model |
| Mistral 7B Instruct | ✅ Works | Mistral (auto) | Well-tested |
| Phi-3 Mini | ✅ Works | ChatML | Lightweight option |

---

## Threading Safety

> **⚠️ TurboQuant + Metal contexts are NOT thread-safe.**

- Contexts must remain on a **single OS thread**
- Do **not** send contexts across threads or use in async executors without pinning
- The Rust wrapper enforces `!Send` + `!Sync` via `PhantomData<*mut ()>`
- In `--chat` mode, inference runs synchronously on the main thread; the token callback redraws the TUI in-place

---

## Architecture

```
Rust (main.rs / tui.rs)
  → ChatTui (tuinix TUI, streaming token callback)
  → ChatSession (message history, KV window management)
  → Template engine (template.rs — ChatML / Llama-3 / Mistral)
  → Safe wrapper (context.rs / TurboQuantCtx)
    → FFI bindings (ffi.rs — API v3, 12 C functions)
      → C shim (llamatqshim.c)
          ├─ mach_absolute_time() instrumentation
          ├─ KV cache ops (llama_memory_seq_rm / seq_add)
          ├─ Streaming token callback (token_cb_fn trampoline)
          └─ llama.cpp TurboQuant fork (C++)
               └─ Metal compute kernels (GPU)
```

---

## Changelog

### [Unreleased] — Multi-Turn Chat TUI (API v3)

Complete multi-turn stateful chat system with interactive TUI, context window management, and chat template auto-detection.

**New features:**
- **`--chat` mode** — Launches a frame-based terminal UI (via `tuinix`) for interactive multi-turn conversation.
- **Live token streaming** — Each generated token triggers a differential frame redraw via a C function-pointer callback trampoline; no terminal corruption, no buffering.
- **Context window management** — Two-strategy KV windowing: primary KV shift (`llama_memory_seq_rm` + `llama_memory_seq_add`) with re-process fallback.
- **Chat template auto-detection** — Reads GGUF metadata to select ChatML, Llama-3, or Mistral Instruct format automatically. Override with `--template`.
- **Default system prompt** — Sensible assistant prompt applied when the model does not specify one.
- **TUI status bar** — Live TPS + TTFT after every response; `/reset`, `/help`, `/stats` slash commands; PgUp/Dn scrollable history.
- **`--template` CLI flag** — Manual override for chat template selection.

**C shim API v3 additions (6 new functions):**
- `llamatq_kv_clear` — Full KV cache clear.
- `llamatq_kv_used` — Query KV cache occupancy via `llama_memory_seq_pos_max`.
- `llamatq_kv_shift` — Sliding-window eviction via `llama_memory_seq_rm` + `llama_memory_seq_add`.
- `llamatq_tokenize` — Tokenize a string into a buffer.
- `llamatq_model_meta` — Read a GGUF metadata key by name.
- `llamatq_chat_eval` — Non-clearing KV append-and-decode with streaming token callback.

**New Rust modules:**
- `src/chat.rs` — `ChatSession`, `ChatMessage`, `WindowConfig`, dual windowing strategies.
- `src/template.rs` — `ChatTemplate` enum with `detect()`, `format_messages()`, system prompt injection.
- `src/tui.rs` — `ChatTui` struct built on `tuinix`; frame rendering, input handling, scroll, notifications.

**Dependencies added:**
- `tuinix = "0.3"` — Lightweight Unix TUI library (only `libc` dependency).

---

### [Unreleased] — LLM Inference Observability

Added standard LLM inference metrics collected entirely inside the C inference loop using `mach_absolute_time()` for nanosecond precision. Metrics are returned to Rust via `llamatq_eval_stats` and displayed after every run.

**Metrics added:** TTFT, Tokens Per Second (generation + prompt), Token Usage (prompt / completion / total), Total Latency.

**Implementation:** API bumped `1 → 2`. `mach_absolute_time()` checkpoints at request start, post-prompt-decode, first-token decode, and generation loop exit. Compact and verbose output modes.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
