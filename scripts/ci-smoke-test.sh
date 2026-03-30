#!/bin/bash
set -e

echo "=== TurboQuant Rust Build Smoke Test ==="
echo "Date: $(date)"
echo "macOS: $(sw_vers -productVersion)"
echo "Xcode: $(xcodebuild -version | head -1)"
echo "Rust: $(rustc --version)"

FORK_COMMIT="a1b2c3d"
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout "$FORK_COMMIT"

mkdir build && cd build
cmake .. -DGGML_METAL=ON -DGGML_USE_TURBOQUANT=1 -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake.log
make -j$(sysctl -n hw.ncpu) llama ggml

grep "GGML_USE_TURBOQUANT: 1" cmake.log || exit 1
grep "CMAKE_BUILD_TYPE: Release" cmake.log || exit 1

cd ../../
export LLAMA_TURBOQUANT_PATH="$(pwd)/llama-cpp-turboquant"
cargo build --release

mkdir -p models
curl -L "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -o models/tinyllama-1.1B.gguf

./target/release/turboquant-llama-rs models/tinyllama-1.1B.gguf "Test"

echo "=== Smoke Test Passed ==="
