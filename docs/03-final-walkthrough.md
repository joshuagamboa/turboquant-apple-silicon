# Walkthrough: TurboQuant Apple Silicon Integration

We have successfully implemented the TurboQuant Rust integration, transitioning from a skeleton structure to a functional inference loop optimized for Apple Silicon.

## Changes Made

### 1. Core C Shim Implementation
- **Migrated to Modern `llama.cpp` API**: Updated `src/shim/llamatqshim.c` to use the latest evaluation and sampling APIs.
- **Configurable Sampling**: Implemented `llamatq_eval_with_sampling` supporting `temperature`, `top_p`, and `seed` via the `llama_sampler_chain` API.
- **Memory Diagnostics**: Added `llamatq_print_memory_breakdown` for inspecting GGML and Metal buffer allocations.
- **KV Cache Reporting**: Fully implemented `llamatq_get_kv_cache_size` using `llama_state_get_size()`, replacing the previous stub.
- **FFI Stability**: Synchronized the C and Rust FFI signatures to ensure memory safety.

### 2. Build System & Linkage
- **Static Linking**: Resolved `dyld` errors on macOS by forcing a static build of all `llama.cpp` and `ggml` components.
- **Linkage Order**: Added all required backends (`ggml-metal`, `ggml-cpu`, `ggml-blas`, `ggml-base`) to the Rust linker configuration.
- **C++ Resolution**: Explicitly linked `libc++` to support the C++ internals of the `llama-cpp-turboquant` fork.

### 3. CLI & Verification
- **Robust CLI**: Integrated `clap` for command-line argument parsing, replacing environment variable lookups.
- **New Options**: Added support for `--temp`, `--top-p`, `--seed`, `--max-tokens`, and `--verbose` (for memory diagnostics).
- **Update Smoke Test**: Updated `scripts/ci-smoke-test.sh` to handle build directory creation, pinned commit checkouts, and environment-aware compilation, while verifying the new CLI flags and non-zero KV cache reporting.
- **End-to-End Success**: The smoke test now pulls a TinyLlama model and successfully generates text using the Metal GPU backend with specified sampling parameters.

## Validation Results

### Build Success
```bash
$ cargo build --release
   Finished release [optimized] target(s) in 47.07s
```

### Static Linking Verification
```bash
$ otool -L target/release/turboquant-llama-rs
target/release/turboquant-llama-rs:
    /System/Library/Frameworks/Metal.framework/Versions/A/Metal
    /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate
    /System/Library/Frameworks/Foundation.framework/Versions/C/Foundation
    /usr/lib/libc++.1.dylib
    ... (No libllama.dylib dependency)
```

### Inference Output
> [!NOTE]
> The Metal backend is successfully initialized and used for all 23 layers of the TinyLlama model.

```text
✓ Metal backend active
...
load_tensors: offloaded 23/23 layers to GPU
...
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
...
✓ Metal backend active
KV Cache Size: 67633152 bytes
API Version: 1
...
[Generated Text with specified sampling]
Generated 256 tokens

=== Memory Breakdown ===
~llama_context:       MTL0 compute buffer size is  66.5000 MiB
~llama_context:        CPU compute buffer size is  12.0137 MiB
...
=== Smoke Test Passed ===
```
