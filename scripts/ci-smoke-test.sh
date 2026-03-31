#!/bin/bash
set -e

echo "=== TurboQuant Rust Build Smoke Test ==="
echo "Date: $(date)"
echo "macOS: $(sw_vers -productVersion)"
echo "Xcode: $(xcodebuild -version | head -1)"
echo "Rust: $(rustc --version)"

FORK_COMMIT="9c600bcd4"
if [ ! -d "llama-cpp-turboquant" ]; then
    git clone https://github.com/TheTom/llama-cpp-turboquant
fi
cd llama-cpp-turboquant
git fetch
git checkout "$FORK_COMMIT"

mkdir -p build && cd build
cmake .. -DGGML_METAL=ON -DGGML_USE_TURBOQUANT=1 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF 2>&1 | tee cmake.log
make -j$(sysctl -n hw.ncpu) llama ggml

cd ../../
export LLAMA_TURBOQUANT_PATH="$(pwd)/llama-cpp-turboquant"
cargo clean
cargo build --release

mkdir -p models
if [ ! -f "models/tinyllama-1.1B.gguf" ]; then
    curl -L "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
      -o models/tinyllama-1.1B.gguf
fi

echo "--- Running test 1: Default Greedy Sampling ---"
./target/release/turboquant-llama-rs models/tinyllama-1.1B.gguf "Test" | tee test1.log
grep "KV Cache Size:" test1.log

if grep -q "0 bytes" test1.log; then
    echo "ERROR: KV Cache size is 0 bytes!"
    exit 1
fi

echo "--- Running test 2: Sampling + Diagnostics ---"
./target/release/turboquant-llama-rs models/tinyllama-1.1B.gguf "Hello" --temp 0.8 --top-p 0.9 --seed 42 --max-tokens 10 --verbose | tee test2.log
grep "=== Memory Breakdown ===" test2.log

echo "=== Smoke Test Passed ==="
