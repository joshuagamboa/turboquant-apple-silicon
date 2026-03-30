// build.rs
use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::fs;

fn main() {
    // Only configure for Apple Silicon (graceful fallback otherwise)
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        
        // Get llama-cpp-turboquant path from env or default
        let llama_path = env::var("LLAMA_TURBOQUANT_PATH")
            .unwrap_or_else(|_| "../llama-cpp-turboquant".to_string());
        
        // Verify path exists
        if !Path::new(&llama_path).exists() {
            panic!(
                "llama-cpp-turboquant not found at '{}'. \
                Set LLAMA_TURBOQUANT_PATH env var or clone the fork.",
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
        cmake.profile("Release");
        
        let dst = cmake.build();
        
        // Link frameworks (MetalKit is optional for compute-only)
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        
        // Find library directory (handles lib/ or lib64/)
        let lib_dir = find_lib_dir(&dst);
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        
        // FIX #1: Link all required libraries (fork may split GGML outputs)
        // Update this list if the fork renames or splits libraries
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-metal");    // Required for Metal kernels
        println!("cargo:rustc-link-lib=static=ggml-base");     // CPU fallback ops
        println!("cargo:rustc-link-lib=static=ggml-blas");     // If Accelerate enabled
        
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
        .include("include")
        // FIX #1: Pass API version via compiler flag (not header macro)
        .define("LLAMA_TURBOQUANT_API_VERSION", "1")
        .flag("-std=c11")
        .flag("-O3")
        .compile("llamatqshim");
    
    println!("cargo:rerun-if-changed=src/shim/llamatqshim.c");
    println!("cargo:rerun-if-changed=include/llamatqshim.h");
}
