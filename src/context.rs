// src/context.rs
use crate::ffi::{
    self, LlamaTqCacheType, LlamaTqCtx, LlamaTqEvalStats, LlamaTqParams, LlamaTqSamplingParams,
    NotSendSync,
};
use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use thiserror::Error;

// ── Sampling parameters ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p:       f32,
    pub seed:        u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.0, top_p: 1.0, seed: 0 }
    }
}

impl From<SamplingParams> for LlamaTqSamplingParams {
    fn from(params: SamplingParams) -> Self {
        Self {
            temperature: params.temperature,
            top_p:       params.top_p,
            seed:        params.seed,
        }
    }
}

// ── Inference statistics ──────────────────────────────────────────────────────

/// Inference statistics returned for every successful eval.
/// All time fields are in milliseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct InferenceStats {
    /// Time from request start to end of first-token decode (ms).
    pub ttft_ms: f64,
    /// Prompt-batch processing time (ms).
    pub prompt_ms: f64,
    /// Generation time (excluding prompt processing, ms).
    pub generation_ms: f64,
    /// Total wall-clock latency (ms).
    pub total_ms: f64,
    /// Number of input (prompt) tokens.
    pub prompt_tokens: i32,
    /// Number of output (generated) tokens.
    pub completion_tokens: i32,
    /// Prompt-processing throughput (tokens/sec).
    pub prompt_tps: f64,
    /// Generation throughput (tokens/sec).
    pub generation_tps: f64,
}

impl From<LlamaTqEvalStats> for InferenceStats {
    fn from(s: LlamaTqEvalStats) -> Self {
        Self {
            ttft_ms:          s.ttft_ms,
            prompt_ms:        s.prompt_ms,
            generation_ms:    s.generation_ms,
            total_ms:         s.total_ms,
            prompt_tokens:    s.prompt_tokens,
            completion_tokens: s.completion_tokens,
            prompt_tps:       s.prompt_tps,
            generation_tps:   s.generation_tps,
        }
    }
}

impl InferenceStats {
    /// Compact one-block output — always shown.
    pub fn print_compact(&self) {
        println!();
        println!("─── Inference Stats ───────────────────────────────");
        println!(
            "  Latency:       {:.1} ms",
            self.total_ms
        );
        println!(
            "  Tokens:        {} prompt → {} generated  ({} total)",
            self.prompt_tokens,
            self.completion_tokens,
            self.prompt_tokens + self.completion_tokens
        );
        println!(
            "  Speed:         {:.1} tok/s  (generation)",
            self.generation_tps
        );
        println!("────────────────────────────────────────────────────");
    }

    /// Detailed output — printed only with --verbose.
    pub fn print_verbose(&self) {
        println!();
        println!("─── Inference Stats (detailed) ─────────────────────");
        println!("  Time to First Token:   {:.1} ms", self.ttft_ms);
        println!(
            "  Prompt Processing:     {:.1} ms  ({:.1} tok/s, {} tokens)",
            self.prompt_ms, self.prompt_tps, self.prompt_tokens
        );
        println!(
            "  Generation:            {:.1} ms  ({:.1} tok/s, {} tokens)",
            self.generation_ms, self.generation_tps, self.completion_tokens
        );
        println!("  Total Latency:         {:.1} ms", self.total_ms);
        println!("  Total Tokens:          {}", self.prompt_tokens + self.completion_tokens);
        println!("─────────────────────────────────────────────────────");
    }
}

impl fmt::Display for InferenceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "latency={:.1}ms ttft={:.1}ms prompt_tok={} completion_tok={} tps={:.1}",
            self.total_ms,
            self.ttft_ms,
            self.prompt_tokens,
            self.completion_tokens,
            self.generation_tps,
        )
    }
}

// ── Error types ───────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum TurboQuantError {
    #[error("Failed to create context: null pointer returned")]
    CreateFailed,
    #[error("Evaluation failed with code {0}: {}", eval_error_description(*.0))]
    EvalFailed(i32),
    #[error("Metal backend not active")]
    MetalNotActive,
    #[error("Invalid UTF-8 in path or prompt")]
    InvalidUtf8(#[from] std::ffi::NulError),
    #[error("API version mismatch: expected {expected}, got {actual}")]
    ApiVersionMismatch { expected: i32, actual: i32 },
    #[error("Context is not thread-safe — do not send across threads")]
    NotThreadSafe,
}

fn eval_error_description(code: i32) -> &'static str {
    match code {
        -1 => "general failure (null argument or context)",
        -2 => "tokenization error (prompt too long or invalid text)",
        -3 => "prompt decode error (llama_decode failed on prompt batch)",
        _  => "unknown error",
    }
}

// ── Safe context wrapper ──────────────────────────────────────────────────────

pub struct TurboQuantCtx {
    raw:     LlamaTqCtx,
    _marker: PhantomData<NotSendSync>,
}

impl TurboQuantCtx {
    /// API version must match the C shim version baked in at compile time.
    const EXPECTED_API_VERSION: i32 = 2;

    /// Create a new TurboQuant context.
    ///
    /// ⚠️ **THREADING**: This context must be used from a single OS thread.
    /// Do not send across threads or use in async executors without pinning.
    pub fn new(
        model_path:   &str,
        n_gpu_layers: i32,
        cache_type:   LlamaTqCacheType,
        n_ctx:        u32,
        n_batch:      u32,
    ) -> Result<Self, TurboQuantError> {
        let api_version = unsafe { ffi::llamatq_get_api_version() };
        if api_version != Self::EXPECTED_API_VERSION {
            return Err(TurboQuantError::ApiVersionMismatch {
                expected: Self::EXPECTED_API_VERSION,
                actual: api_version,
            });
        }

        let model_path_c = CString::new(model_path)?;

        let params = LlamaTqParams {
            model_path:   model_path_c.as_ptr(),
            n_gpu_layers,
            cache_type_k: cache_type,
            cache_type_v: cache_type,
            n_ctx,
            n_batch,
        };

        let raw = unsafe { ffi::llamatq_create(&params) };

        if raw.is_null() {
            return Err(TurboQuantError::CreateFailed);
        }

        Ok(Self { raw, _marker: PhantomData })
    }

    pub fn is_metal_active(&self) -> bool {
        unsafe { ffi::llamatq_is_metal_active(self.raw) != 0 }
    }

    pub fn kv_cache_size(&self) -> usize {
        unsafe { ffi::llamatq_get_kv_cache_size(self.raw) }
    }

    /// Evaluate the prompt with explicit sampling parameters.
    /// Always collects inference statistics.
    ///
    /// ⚠️ Must be called from the same thread that created the context.
    pub fn eval_with_sampling(
        &self,
        prompt:     &str,
        max_tokens: i32,
        params:     SamplingParams,
    ) -> Result<InferenceStats, TurboQuantError> {
        let prompt_c  = CString::new(prompt)?;
        let c_params: LlamaTqSamplingParams = params.into();
        let mut raw_stats = LlamaTqEvalStats::default();

        let result = unsafe {
            ffi::llamatq_eval_with_sampling(
                self.raw,
                prompt_c.as_ptr(),
                max_tokens,
                &c_params,
                &mut raw_stats,
            )
        };

        if result < 0 {
            Err(TurboQuantError::EvalFailed(result))
        } else {
            Ok(InferenceStats::from(raw_stats))
        }
    }

    /// Evaluate the prompt with default (greedy) sampling.
    /// Always collects inference statistics.
    ///
    /// ⚠️ Must be called from the same thread that created the context.
    pub fn eval(&self, prompt: &str, max_tokens: i32) -> Result<InferenceStats, TurboQuantError> {
        self.eval_with_sampling(prompt, max_tokens, SamplingParams::default())
    }

    pub fn print_memory_breakdown(&self) {
        unsafe { ffi::llamatq_print_memory_breakdown(self.raw) }
    }

    pub fn verify_metal(&self) -> Result<(), TurboQuantError> {
        if self.is_metal_active() {
            Ok(())
        } else {
            Err(TurboQuantError::MetalNotActive)
        }
    }

    pub fn as_raw(&self) -> LlamaTqCtx {
        self.raw
    }
}

impl Drop for TurboQuantCtx {
    fn drop(&mut self) {
        unsafe { ffi::llamatq_free(self.raw) }
    }
}
