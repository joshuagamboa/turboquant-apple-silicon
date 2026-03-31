mod ffi;
mod context;
mod template;
mod chat;
mod tui;

use clap::Parser;
use context::{SamplingParams, TurboQuantCtx};
use ffi::LlamaTqCacheType;
use chat::{ChatSession, WindowConfig};
use tui::ChatTui;

#[derive(Parser, Debug)]
#[command(author, version, about = "TurboQuant Apple Silicon Inference Native Wrapper")]
struct Args {
    /// Path to the GGUF model file
    #[arg(required = true)]
    model: String,

    /// Prompt to evaluate (one-shot mode only; ignored in --chat)
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

    /// Maximum number of tokens to generate per turn
    #[arg(long, default_value_t = 4096)]
    max_tokens: i32,

    /// Context size (number of tokens)
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,

    /// Batch size for prompt processing
    #[arg(long, default_value_t = 512)]
    batch_size: u32,

    /// Print detailed diagnostics: memory breakdown + full inference stats
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Launch interactive multi-turn chat TUI
    #[arg(long, default_value_t = false)]
    chat: bool,

    /// Override chat template (chatml | llama3 | mistral); auto-detected if omitted
    #[arg(long)]
    template: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Session logging & crash capture for TUI
    if args.chat {
        let _ = context::init_logging("turboquant.log");
        
        std::panic::set_hook(Box::new(|info| {
            let payload = info.payload().downcast_ref::<&str>().copied()
                .or_else(|| info.payload().downcast_ref::<String>().map(|s| s.as_str()))
                .unwrap_or("unknown panic");
            let location = info.location().map(|l| format!(" at {}:{}", l.file(), l.line()))
                .unwrap_or_default();
            
            let msg = format!("\n\n!!! RUST PANIC !!!\nMessage: {}{}\n", payload, location);
            
            // Try to log to file
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("turboquant.log") {
                use std::io::Write;
                let _ = writeln!(f, "{}", msg);
            }
            
            // Still print to stderr as a last resort
            eprintln!("{}", msg);
            
            // Safety measure to ensure auto-wrap is re-enabled if panicked in TUI
            print!("\x1b[?7h");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }));

        eprintln!("Loading model into TurboQuant Engine... (this may take a moment)");
    }

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
        Ok(())  => eprintln!("✓ Metal backend active"),
        Err(e) => {
            eprintln!("⚠️  Warning: {}", e);
            eprintln!("   Ensure GGML_METAL=ON and running on Apple Silicon");
        }
    }

    // ── Verbose pre-run diagnostics ───────────────────────────────────────────
    if args.verbose {
        let kv_size = ctx.kv_cache_size();
        eprintln!(
            "KV Cache Size: {} bytes ({:.2} MB)",
            kv_size,
            kv_size as f64 / 1024.0 / 1024.0
        );
        eprintln!("API Version:   {}", unsafe { ffi::llamatq_get_api_version() });
        eprintln!("=== Memory Breakdown ===");
        ctx.print_memory_breakdown();
        eprintln!("========================");
    }

    let sparams = SamplingParams {
        temperature: args.temp,
        top_p:       args.top_p,
        seed:        args.seed,
    };

    // ── Chat TUI mode ─────────────────────────────────────────────────────────
    if args.chat {
        let win_config = WindowConfig {
            ctx_size:         args.ctx_size as i32,
            keep_ratio:       0.5,
            response_reserve: args.max_tokens.max(1024),
        };

        let mut session = if let Some(ref tmpl_str) = args.template {
            use template::ChatTemplate;
            let tmpl = match tmpl_str.to_lowercase().as_str() {
                "chatml"  => ChatTemplate::ChatML,
                "llama3"  => ChatTemplate::Llama3,
                "mistral" => ChatTemplate::MistralInstruct,
                other => {
                    eprintln!("⚠️  Unknown template '{}'; auto-detecting.", other);
                    ChatTemplate::detect(&ctx)
                }
            };
            eprintln!("Ä Template: {}", tmpl);
            ChatSession::with_template(tmpl, win_config)
        } else {
            let s = ChatSession::new(&ctx, win_config);
            eprintln!("✓ Auto-detected template: {}", s.template_name());
            s
        };

        // Extract the model file name for display in the TUI header
        let model_display = std::path::Path::new(&args.model)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&args.model)
            .to_string();

        let tui = match ChatTui::new(
            &mut session,
            &ctx,
            sparams,
            args.max_tokens,
            model_display,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("✗ Failed to initialise TUI: {}", e);
                std::process::exit(1);
            }
        };

        if let Err(e) = tui.run() {
            eprintln!("✗ TUI error: {}", e);
            std::process::exit(1);
        }

        return; // clean exit from chat mode
    }

    // ── One-shot inference (original behaviour) ───────────────────────────────
    let stats = match ctx.eval_with_sampling(&args.prompt, args.max_tokens, sparams) {
        Ok(stats) => stats,
        Err(e)    => {
            eprintln!("✗ Inference error: {}", e);
            std::process::exit(1);
        }
    };

    // ── Stats output ──────────────────────────────────────────────────────────
    if args.verbose {
        stats.print_verbose();
    } else {
        stats.print_compact();
    }
}
