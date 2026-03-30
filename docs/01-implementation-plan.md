# Fix Linkage and Complete Verification

The current build produces dynamic libraries (`.dylib`) which cause `dyld` errors at runtime because the Rust binary's `rpath` is not correctly configured to find them. To ensure a robust, "production-grade" integration template, we will move to static linking for the internal dependencies.

## Proposed Changes

### [Component] Build System

#### [MODIFY] [build.rs](file:///Users/jpg/Coding/turboquant-apple-silicon/build.rs)
- Explicitly disable shared library builds in the `cmake` configuration.
- Add `cmake.define("BUILD_SHARED_LIBS", "OFF");`.
- Add a cleanup step to remove any existing `.dylib` files in the `OUT_DIR` to prevent the linker from picking them up by accident.

#### [MODIFY] [ci-smoke-test.sh](file:///Users/jpg/Coding/turboquant-apple-silicon/scripts/ci-smoke-test.sh)
- Update the manual `cmake` call to include `-DBUILD_SHARED_LIBS=OFF`.
- Add `cargo clean` before the final build to ensure a fresh link against static libraries.

## Verification Plan

### Automated Tests
- Run `scripts/ci-smoke-test.sh`. 
- This will:
  1. Clean the build environment.
  2. Rebuild `llama-cpp-turboquant` as static libraries.
  3. Rebuild the Rust project linking against these static libraries.
  4. Download the TinyLlama model.
  5. Execute the binary and verify it generates text without `dyld` errors.

### Manual Verification
- Verify that `ls llama-cpp-turboquant/build` contains `.a` files instead of `.dylib`.
- Run `./target/release/turboquant-llama-rs models/tinyllama-1.1B.gguf "Hello"` manually to confirm streaming output works.
- Run `otool -L target/release/turboquant-llama-rs` to confirm NO dynamic dependency on `libllama.dylib`.

## Open Questions
- None. Static linking is the standard approach for this type of integration template to avoid deployment friction.
