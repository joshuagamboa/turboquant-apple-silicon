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
  --temp 0.7 --top-p 0.9 --max-tokens 4096
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
| `--max-tokens` | `4096` | Maximum tokens to generate per turn (raised for long responses). |
| `--ctx-size` | `8192` | Total context window size in tokens. |
| `--batch-size` | `512` | Prompt-processing batch size in tokens. |
| `--verbose` | (Flag) | Print detailed diagnostics: memory breakdown, TTFT, per-phase timing. |

---

## Interactive Chat TUI

Launch with `--chat` to enter a high-performance terminal UI:

```
 󱚣 TurboQuant Chat │ Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf │ ChatML │ 312/8192 tokens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  You › Who wrote Don Quixote?

   AI › <think>
        The user is asking about the authorship of Don Quixote.
        </think>
        Miguel de Cervantes wrote Don Quixote, published in two parts in 1605
        and 1615. It is widely regarded as the first modern novel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❯ _
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 45.2 tok/s  TTFT 83ms │ Ctrl+C: quit  |  /reset  |  /help  |  PgUp/Dn: scroll
```

### Modern Features
- **Vibrant UI:** Refreshed color palette with high-contrast text and sleek separators.
- **Thinking Mode:** Internal model reasoning (between `<think>` tags) is styled in dimmed italics for better visual separation.
- **Advanced Trackpad Support:** Native two-finger scrolling optimized for MacBook Pro (SGR 1006 + Motion 1002 modes).
- **Sticky-Bottom Auto-scroll:** Viewport intelligently follows new tokens during streaming but allows you to scroll up manually without being "yanked" back.

### TUI Navigation

| Key | Action |
|-----|--------|
| `Enter` | Submit message / run inference |
| `Ctrl+C` | Exit chat |
| `←` / `→` | Move cursor in input |
| `Ctrl+U` / `Ctrl+B` | Scroll Up (MacBook friendly) |
| `Ctrl+D` / `Ctrl+F` | Scroll Down (MacBook friendly) |
| `PgUp` / `PgDn` | Scroll history |
| `Home` / `End` | Jump to start / end of input |
| `Backspace` / `Del` | Delete character |

---

## Context Window Management

Long conversations eventually exceed the model's context window. TurboQuant implements a sophisticated windowing system:

### Smart KV Eviction
Unlike standard circular buffers, TurboQuant's **KV Shift** strategy preserves the **System Prompt** (Position 0..X) while sliding the middle context window. This ensures the AI maintains its persona and instructions even during extremely long sessions.

- **High Water Mark:** Windowing triggers when context is nearly full.
- **Response Reserve:** 1024 tokens are always reserved for the assistant's next turn to prevent truncation.
- **System Preservation:** System instructions are pinned and never evicted during KV shifts.

---

## Chat Template Auto-Detection

The template engine reads GGUF metadata keys (`tokenizer.chat_template`, `general.architecture`) to automatically identify and apply the correct chat format:

| Template | Detected From | Format |
|----------|--------------|--------|
| **ChatML** | `tokenizer.chat_template` contains `im_start`, or Qwen architecture | `<|im_start|>system…<|im_end|>` |
| **Llama-3** | Architecture `llama` or template mentions `<|start_header_id|>` | `<|start_header_id|>user<|end_header_id|>…` |
| **Mistral Instruct** | Architecture `mistral` or template mentions `[INST]` | `[INST] … [/INST]` |

Override with `--template chatml|llama3|mistral` if auto-detection is wrong.

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

### Metrics Reference

| Metric | Where measured | Description |
|---|---|---|
| **TTFT** | C shim (`mach_absolute_time`) | Time from request start to end of first-token decode |
| **Latency** | C shim | Total wall-clock time (prompt processing + generation) |
| **Prompt TPS** | C shim | Prompt tokens processed per second |
| **Generation TPS** | C shim | Output tokens generated per second |
| **Token Usage** | C shim | Input prompt token count + output completion token count |
| **Context Used** | KV cache query | Current KV cache occupancy (shown in TUI header) |

---

## Threading Safety

> **⚠️ TurboQuant + Metal contexts are NOT thread-safe.**

- Contexts must remain on a **single OS thread**
- Do **not** send contexts across threads or use in async executors without pinning
- The Rust wrapper enforces `!Send` + `!Sync` via `PhantomData<*mut ()>`

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
