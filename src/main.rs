// src/main.rs
mod ffi;
mod context;

use context::{TurboQuantCtx, TurboQuantError};
use ffi::LlamaTqCacheType;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [prompt]", args[0]);
        eprintln!();
        eprintln!("⚠️ THREADING: Context must stay on single OS thread");
        std::process::exit(1);
    }
    
    let model_path = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello, world!");
    
    let ctx = match TurboQuantCtx::new(
        model_path,
        99,
        LlamaTqCacheType::Turbo3,
        2048,
        512,
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create context: {}", e);
            std::process::exit(1);
        }
    };
    
    match ctx.verify_metal() {
        Ok(()) => println!("✓ Metal backend active"),
        Err(e) => {
            eprintln!("⚠️ Warning: {}", e);
            eprintln!("   Ensure GGML_METAL=ON and running on Apple Silicon");
        }
    }
    
    println!("KV Cache Size: {} bytes", ctx.kv_cache_size());
    println!("API Version: {}", unsafe { ffi::llamatq_get_api_version() });
    
    match ctx.eval(prompt, 256) {
        Ok(tokens) => {
            println!("\nGenerated {} tokens", tokens);
        },
        Err(e) => {
            eprintln!("Evaluation failed: {}", e);
            std::process::exit(1);
        }
    }
}
