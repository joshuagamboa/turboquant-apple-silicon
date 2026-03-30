# turboquant-apple-silicon

**TurboQuant KV Cache Quantization for llama.cpp on Apple Silicon (Metal/ARM)**

A production-grade Rust integration template that wraps the [TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant) of llama.cpp, enabling aggressive KV cache compression with minimal quality loss — optimised for Apple Silicon GPUs via Metal compute shaders.

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
│   ├── main.rs                 # CLI entry point
│   ├── ffi.rs                  # Hand-written C FFI bindings
│   ├── context.rs              # Safe Rust wrapper (TurboQuantCtx)
│   └── shim/
│       └── llamatqshim.c       # C shim bridging Rust ↔ llama.cpp
├── include/
│   └── llamatqshim.h           # C shim header
├── llama-cpp-turboquant/       # Fork (submodule or symlink)
├── scripts/
│   └── ci-smoke-test.sh        # CI smoke test
├── models/
│   └── .gitkeep                # GGUF models go here (gitignored)
└── README.md
```

---

## Quick Start

### 1. Clone the TurboQuant fork

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout a1b2c3d   # Pinned commit
cd ..
```

### 2. Build

```bash
export LLAMA_TURBOQUANT_PATH="$(pwd)/llama-cpp-turboquant"
cargo build --release
```

### 3. Run

```bash
./target/release/turboquant-llama-rs models/your-model.gguf "Hello, world!"
```

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
  → Safe wrapper (context.rs)
    → FFI bindings (ffi.rs)
      → C shim (llamatqshim.c)
        → llama.cpp TurboQuant fork (C++)
          → Metal compute kernels (GPU)
```

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
