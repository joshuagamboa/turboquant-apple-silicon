# turboquant-apple-silicon

**High-Performance Rust Integration for KV Cache Quantized LLM Inference on Apple Silicon**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-lightgrey.svg)](https://developer.apple.com/apple-silicon/)
[![Status: Functional](https://img.shields.io/badge/Status-Fully%20Functional-brightgreen.svg)]()

`turboquant-apple-silicon` is a production-grade Rust integration that brings **aggressive KV cache quantization** to Apple Silicon GPUs (M1/M2/M3/M4/M5). By wrapping a specialized fork of `llama.cpp`, this project enables significantly reduced memory footprints for large models and long-context windows without sacrificing the power of Metal acceleration.

---

## 🚀 What makes this project unique?

This integration bridges the gap between low-level C++ performance and high-level Rust safety and ergonomics. It features:

*   **⚡ Metal-Optimized Kernels**: Direct GPU acceleration for quantized KV cache operations.
*   **📉 Massive Memory Savings**: Up to **6.4× compression** of the KV cache using `turbo2` and `turbo3` quantization levels.
*   **💬 Interactive TUI**: A sleek, terminal-based chat interface with "thinking mode" styling and trackpad support.
*   **🧠 Smart Context Management**: Sophisticated **KV Shift** strategy that preserves the system prompt while sliding the conversation window.
*   **🔍 Full Observability**: Real-time metrics for Time-to-First-Token (TTFT), Tokens Per Second (TPS), and memory allocation.

---

## 🔬 How It Works

### High-Level Architecture

```mermaid
flowchart TD
    subgraph User["User Interface"]
        CLI["CLI Mode\n(one-shot prompt)"]
        TUI["Interactive TUI\n(multi-turn chat)"]
    end

    subgraph Rust["Rust Application"]
        MAIN["main.rs\nCLI parsing & mode dispatch"]
        TUIMOD["tui.rs\nTerminal UI, event loop,\nreal-time token streaming"]
        CHAT["chat.rs\nChatSession, message history,\nKV windowing strategies"]
        TMPL["template.rs\nAuto-detect & format\nChatML / Llama3 / Mistral"]
        CTX["context.rs\nTurboQuantCtx\n(safe Rust wrapper)"]
        FFI["ffi.rs\nUnsafe extern C declarations"]
    end

    subgraph C["C Shim Layer"]
        SHIM["llamatqshim.c\nGlue layer wrapping llama.cpp API\n(context, eval, KV ops, tokenize)"]
    end

    subgraph TQ["llama-cpp-turboquant (static libs)"]
        LLAMA["libllama.a\nModel loading, decoding,\nsampling"]
        GGML["libggml.a + libggml-base.a\nTensor operations"]
        METAL["libggml-metal.a\nMetal GPU kernels incl.\nTurboQuant quantization"]
        CPU["libggml-cpu.a + libggml-blas.a\nCPU fallback paths"]
    end

    subgraph HW["Apple Silicon Hardware"]
        GPU["Metal GPU\n(quantized KV cache ops)"]
        ACC["Accelerate Framework\n(CPU BLAS)"]
    end

    CLI --> MAIN
    TUI --> MAIN
    MAIN -->|"--chat"| TUIMOD
    MAIN -->|"one-shot"| CTX
    TUIMOD --> CHAT
    CHAT --> TMPL
    CHAT --> CTX
    CTX --> FFI
    FFI -->|"unsafe FFI"| SHIM
    SHIM --> LLAMA
    LLAMA --> GGML
    GGML --> METAL
    GGML --> CPU
    METAL --> GPU
    CPU --> ACC
```

### How TurboQuant Is Used

**TurboQuant** is a specialized fork of `llama.cpp` that adds custom Metal GPU compute kernels for **KV cache quantization**. In standard llama.cpp, the Key and Value caches used during attention are stored in FP16, consuming significant GPU memory. TurboQuant introduces two new quantization types — `Turbo2` and `Turbo3` — that compress these caches at the hardware level using optimized Metal shaders.

This project harnesses TurboQuant through a multi-layer integration:

1. **Build time** (`build.rs`): CMake compiles the TurboQuant fork with `GGML_METAL=ON` and `GGML_USE_TURBOQUANT=1`, producing static libraries including the quantized Metal kernels.
2. **FFI boundary** (`ffi.rs`): Defines `LlamaTqCacheType::Turbo3` (and `Turbo2`) as Rust enum variants that map directly to the fork's `ggml_type` enum values.
3. **Context creation** (`context.rs` → `llamatqshim.c`): When `TurboQuantCtx::new()` is called, it passes the cache type through FFI to the C shim, which sets `ctx_params.type_k` and `ctx_params.type_v` to the TurboQuant type. This single assignment is what activates the quantized Metal kernels for all subsequent KV cache operations.
4. **Runtime inference**: Every `llama_decode()` call during token generation now stores and retrieves KV entries using the quantized format on the GPU — no application-level code changes needed beyond the initial configuration.

The result: **up to 6.4x KV cache compression** with Metal-accelerated quantization/dequantization, enabling longer context windows and larger models within the same memory budget.

### TurboQuant Integration Close-Up

```mermaid
flowchart TD
    subgraph Init["Context Initialization"]
        A["main.rs\nTurboQuantCtx::new(\n  model, 99, Turbo3,\n  ctx_size, batch_size\n)"]
        B["context.rs\nBuilds LlamaTqParams {\n  cache_type_k: Turbo3,\n  cache_type_v: Turbo3\n}"]
        C["ffi.rs\nllamatq_create(&params)\n(unsafe extern C)"]
        D["llamatqshim.c\nctx_params.type_k =\n  (ggml_type)params->cache_type_k\nctx_params.type_v =\n  (ggml_type)params->cache_type_v"]
        E["llama_init_from_model()\nAllocates KV cache buffers\nin Turbo3 format on Metal GPU"]
    end

    subgraph Inference["Chat Inference Loop"]
        F["chat.rs :: ChatSession::send()\nFormats turn via template.rs"]
        G["context.rs :: chat_eval()\nCalls ffi::llamatq_chat_eval()"]
        H["llamatqshim.c :: llamatq_chat_eval()\n1. Tokenize formatted turn\n2. llama_decode() → KV cache\n3. Sample loop with token_cb\n4. Return stats (TTFT, TPS)"]
        I["Metal GPU\nAttention reads/writes KV\nusing Turbo3 quantized kernels"]
    end

    subgraph KV["KV Cache Management"]
        J["chat.rs :: apply_windowing()\nTriggered when context fills"]
        K["context.rs :: kv_shift()\n→ llamatqshim.c:\n  llama_memory_seq_rm(p0..p1)\n  llama_memory_seq_add(shift)"]
        L["context.rs :: kv_clear()\n→ llamatqshim.c:\n  llama_kv_self_clear()\n(fallback: full reprocess)"]
    end

    A --> B --> C --> D --> E
    F --> G --> H --> I
    J -->|"primary: shift"| K
    J -->|"fallback: reprocess"| L
    H -.->|"KV cache ops use\nTurbo3 Metal kernels"| I
    K -.->|"operates on quantized\nKV cache in-place"| I
```

---

## 🏗️ Open Source Foundations

This project stands on the shoulders of giants:

1.  **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: The industry-standard implementation for efficient LLM inference.
2.  **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)**: A specialized fork providing the **TurboQuant** Metal compute kernels for KV cache quantization.
3.  **[tuinix](https://github.com/jpg/tuinix)**: The frame-based TUI engine used for the interactive chat interface.

---

## 🛠️ Installation & Setup

### Prerequisites

*   **macOS**: 13.0 (Ventura) or newer.
*   **Apple Silicon**: M-series chip (M1, M2, M3, M4, etc.).
*   **Tools**: Latest stable **Rust**, **CMake** (3.28+), and **Xcode Command Line Tools**.

### 1. Clone & Prepare the Submodule

The TurboQuant fork is required and should be cloned into the project root.

```bash
git clone https://github.com/jpg/turboquant-apple-silicon.git
cd turboquant-apple-silicon

# Clone the required llama-cpp-turboquant fork
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout 9c600bcd4   # Pinned stable commit for this integration
cd ..
```

### 2. Build the Project

The build system automatically handles the CMake configuration and **statically links** the `llama.cpp` internals to ensure a portable, single-binary output.

```bash
cargo build --release
```

### 3. Model Management

This project uses standard `.gguf` model files. You should place your models in the `models/` directory (created automatically or manually).

```bash
mkdir -p models
# Download your favorite model (e.g., Llama-3, Qwen, Mistral) into models/
```

---

## 📖 Usage Guide

The binary `turboquant-llama-rs` supports two primary modes: **One-shot CLI** and **Interactive TUI Chat**.

### A. CLI Mode (One-Shot Inference)
Best for scripts, automated tests, or single-prompt queries.

```bash
./target/release/turboquant-llama-rs models/llama-3-8b.gguf "Why is the sky blue?" \
  --temp 0.7 \
  --top-p 0.9 \
  --max-tokens 512
```

### B. Interactive TUI Mode (Multi-turn Chat)
Enter a high-performance terminal UI designed for multi-turn conversations.

```bash
./target/release/turboquant-llama-rs models/llama-3-8b.gguf --chat
```

**TUI Controls:**
*   **`Enter`**: Submit message.
*   **`Ctrl+C`**: Quit.
*   **`PgUp` / `PgDn`**: Scroll chat history.
*   **Trackpad**: Native two-finger scrolling supported.
*   **`/reset`**: Clear the current conversation context.

---

## ⚙️ Configuration Flags

| Flag | Default | Description |
|---|---|---|
| `<model_path>` | (Required) | Path to your GGUF model file. |
| `--chat` | `false` | Enable the interactive TUI mode. |
| `--ctx-size` | `8192` | Total context window size in tokens. |
| `--temp` | `0.0` | Sampling temperature (0.0 = greedy). |
| `--top-p` | `1.0` | Nucleus sampling (1.0 = disabled). |
| `--seed` | `0` | RNG seed for deterministic outputs. |
| `--max-tokens` | `4096` | Max tokens to generate per response. |
| `--verbose` | `false` | Print detailed memory diagnostics and timing data. |
| `--template` | (Auto) | Override chat template (`chatml`, `llama3`, `mistral`). |

---

## 🧩 Advanced Features

### Chat Template Auto-Detection
The engine automatically identifies the correct prompt format (ChatML, Llama-3, Mistral) by inspecting GGUF metadata. You rarely need to specify this manually unless using a non-standard fine-tune.

### KV Shift Windowing
To handle long conversations, TurboQuant implements a **KV Shift** strategy. Unlike simple circular buffers, it **pins the System Prompt** at the beginning of the context and only shifts the conversational "middle ground." This ensures the model never "forgets" its core instructions.

### Thinking Mode Styling
Models with internal reasoning (like DeepSeek or R1) often output `<think>...</think>` tags. The TUI automatically detects these and styles them in a **dimmed, italicized** font to visually separate reasoning from the final answer.

---

## ⚠️ Development Notes

*   **Static Linking**: The project forces static linkage of `ggml` and `llama` libraries to avoid `dyld` path issues common on macOS.
*   **Threading**: Metal contexts are **not thread-safe**. The Rust wrapper enforces `!Send` and `!Sync` to prevent safety violations across thread boundaries.
*   **Memory Reporting**: Run with `--verbose` to see exact buffer sizes allocated on the Metal GPU vs. System CPU.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
