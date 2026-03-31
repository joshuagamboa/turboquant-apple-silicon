// build.rs
use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::fs;

fn main() {
    // Only configure for Apple Silicon (graceful fallback otherwise)
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        
        // Get llama-cpp-turboquant path from env or default (in-tree)
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let llama_path = env::var("LLAMA_TURBOQUANT_PATH").unwrap_or_else(|_| {
            Path::new(&manifest_dir)
                .join("llama-cpp-turboquant")
                .to_string_lossy()
                .to_string()
        });
        
        // Verify path exists
        if !Path::new(&llama_path).exists() {
            panic!(
                "llama-cpp-turboquant not found at '{}'. \
                Clone the fork into the project root:\n  \
                git clone https://github.com/TheTom/llama-cpp-turboquant\n\
                Or set LLAMA_TURBOQUANT_PATH env var to an external checkout.",
                llama_path
            );
        }
        
        let mut cmake = Config::new(&llama_path);
        
        // Metal support (standard flag)
        cmake.define("GGML_METAL", "ON");
        
        // TurboQuant (FORK-SPECIFIC)
        cmake.define("GGML_USE_TURBOQUANT", "1");
        
        // Apple Accelerate framework (optional)
        cmake.define("GGML_USE_ACCELERATE", "1");
        
        // Architecture (explicit for Apple Silicon)
        cmake.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        
        // FIX #5: Explicitly set Release build type (Debug is 10x slower for Metal)
        cmake.define("CMAKE_BUILD_TYPE", "Release");
        cmake.define("BUILD_SHARED_LIBS", "OFF");
        cmake.profile("Release");
        
        let dst = cmake.build();
        let lib_dir = find_lib_dir(&dst);
        
        // FIX #6: Ensure no dynamic libraries are present in the search path to prevent dyld errors
        if let Ok(entries) = std::fs::read_dir(&lib_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "dylib" {
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }
        
        // Link frameworks (MetalKit is optional for compute-only)
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        
        // Find library directory (handles lib/ or lib64/)
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        
        // FIX #1: Link all required libraries (fork may split GGML outputs)
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-metal");    // Required for Metal kernels
        println!("cargo:rustc-link-lib=static=ggml-base");     // CPU fallback ops
        println!("cargo:rustc-link-lib=static=ggml-blas");     // If Accelerate enabled
        println!("cargo:rustc-link-lib=static=ggml-cpu");      // Core CPU ops
        // FIX #7: Link C++ standard library (required for static llama.cpp)
        println!("cargo:rustc-link-lib=c++");
        
        // FIX #2: Build the shim explicitly using cc crate
        build_shim(&llama_path);
        
        // Rebuild if C files change
        println!("cargo:rerun-if-changed=include/llamatqshim.h");
        println!("cargo:rerun-if-changed=src/shim/llamatqshim.c");
        println!("cargo:rerun-if-env-changed=LLAMA_TURBOQUANT_PATH");
        
        // Copy headers for reference
        let out_dir = env::var("OUT_DIR").unwrap();
        fs::create_dir_all(format!("{}/include", out_dir)).unwrap();
        fs::copy("include/llamatqshim.h", format!("{}/include/llamatqshim.h", out_dir))
            .expect("Failed to copy shim header");
        
    } else {
        println!("cargo:warning=This build is optimized for Apple Silicon (aarch64-macos)");
        println!("cargo:warning=TurboQuant Metal support requires macOS + ARM");
    }
}

fn find_lib_dir(dst: &Path) -> PathBuf {
    // Check common lib directories
    for subdir in ["lib", "lib64", "build/lib", "build/lib64"] {
        let path = dst.join(subdir);
        if path.exists() {
            return path;
        }
    }
    // Fallback to dst itself
    dst.to_path_buf()
}

fn build_shim(llama_path: &str) {
    cc::Build::new()
        .file("src/shim/llamatqshim.c")
        .include(format!("{}/include", llama_path))
        .include(format!("{}/ggml/include", llama_path))
        .include("include")
        // FIX #1: Pass API version via compiler flag (not header macro)
        .define("LLAMA_TURBOQUANT_API_VERSION", "3")
        .flag("-std=c11")
        .flag("-O3")
        .compile("llamatqshim");
    
    println!("cargo:rerun-if-changed=src/shim/llamatqshim.c");
    println!("cargo:rerun-if-changed=include/llamatqshim.h");
}
