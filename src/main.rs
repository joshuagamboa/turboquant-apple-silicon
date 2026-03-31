mod ffi;
mod context;

use clap::Parser;
use context::{SamplingParams, TurboQuantCtx};
use ffi::LlamaTqCacheType;

#[derive(Parser, Debug)]
#[command(author, version, about = "TurboQuant Apple Silicon Inference Native Wrapper")]
struct Args {
    /// Path to the GGUF model file
    #[arg(required = true)]
    model: String,

    /// Prompt to evaluate
    #[arg(default_value = "Hello, world!")]
    prompt: String,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temp: f32,

    /// Top-p (nucleus) sampling (1.0 = disabled)
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// RNG seed for reproducibility
    #[arg(long, default_value_t = 0)]
    seed: u32,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: i32,

    /// Context size (number of tokens)
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,

    /// Batch size for prompt processing
    #[arg(long, default_value_t = 512)]
    batch_size: u32,

    /// Print detailed diagnostics: memory breakdown + full inference stats (TTFT, prompt TPS, etc.)
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    // ── Context creation ──────────────────────────────────────────────────────
    let ctx = match TurboQuantCtx::new(
        &args.model,
        99,
        LlamaTqCacheType::Turbo3,
        args.ctx_size,
        args.batch_size,
    ) {
        Ok(ctx)  => ctx,
        Err(e)   => {
            eprintln!("✗ Failed to create context: {}", e);
            std::process::exit(1);
        }
    };

    // ── Backend check ─────────────────────────────────────────────────────────
    match ctx.verify_metal() {
        Ok(())  => println!("✓ Metal backend active"),
        Err(e) => {
            eprintln!("⚠️  Warning: {}", e);
            eprintln!("   Ensure GGML_METAL=ON and running on Apple Silicon");
        }
    }

    // ── Verbose pre-run diagnostics ───────────────────────────────────────────
    if args.verbose {
        let kv_size = ctx.kv_cache_size();
        println!(
            "KV Cache Size: {} bytes ({:.2} MB)",
            kv_size,
            kv_size as f64 / 1024.0 / 1024.0
        );
        println!("API Version:   {}", unsafe { ffi::llamatq_get_api_version() });
        println!("=== Memory Breakdown ===");
        ctx.print_memory_breakdown();
        println!("========================");
    }

    // ── Inference ─────────────────────────────────────────────────────────────
    let sparams = SamplingParams {
        temperature: args.temp,
        top_p:       args.top_p,
        seed:        args.seed,
    };

    let stats = match ctx.eval_with_sampling(&args.prompt, args.max_tokens, sparams) {
        Ok(stats) => stats,
        Err(e)    => {
            eprintln!("✗ Inference error: {}", e);
            std::process::exit(1);
        }
    };

    // ── Stats output ──────────────────────────────────────────────────────────
    if args.verbose {
        // Full detail: TTFT, prompt processing speed, per-phase timing.
        stats.print_verbose();
    } else {
        // Always-on compact summary: latency, token counts, generation TPS.
        stats.print_compact();
    }
}
