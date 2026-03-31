// src/context.rs
use crate::ffi::{
    self, LlamaTqCacheType, LlamaTqCtx, LlamaTqEvalStats, LlamaTqParams, LlamaTqSamplingParams,
    NotSendSync,
};
use libc::{c_char, c_void};
use std::ffi::{CStr, CString};
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

pub fn disable_logs() {
    unsafe { ffi::llamatq_disable_logs() }
}

/// Initialize file logging for backend diagnostics.
pub fn init_logging(path: &str) -> std::io::Result<()> {
    let c_path = std::ffi::CString::new(path).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, e)
    })?;
    unsafe { ffi::llamatq_init_logging(c_path.as_ptr()) };
    Ok(())
}

pub struct TurboQuantCtx {
    raw:     LlamaTqCtx,
    _marker: PhantomData<NotSendSync>,
}

impl TurboQuantCtx {
    /// API version must match the C shim version baked in at compile time.
    const EXPECTED_API_VERSION: i32 = 3;

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

    // ── API v3: KV cache management ───────────────────────────────────────────

    /// Clear the KV cache. Call before re-processing the full conversation
    /// history in the Re-process windowing strategy.
    pub fn kv_clear(&self) {
        unsafe { ffi::llamatq_kv_clear(self.raw) }
    }

    /// Return the number of tokens currently stored in the KV cache.
    pub fn kv_used(&self) -> i32 {
        unsafe { ffi::llamatq_kv_used(self.raw) }
    }

    /// Remove KV positions `[p0, p1)` and shift remaining positions down.
    /// Used by the KV Shift windowing strategy to evict old turns in-place.
    pub fn kv_shift(&self, p0: i32, p1: i32) {
        unsafe { ffi::llamatq_kv_shift(self.raw, p0, p1) }
    }

    // ── API v3: Tokenization ──────────────────────────────────────────────────

    /// Tokenize `text` and return token IDs. Does NOT decode into the KV cache.
    /// `add_special`: if true, includes BOS/EOS special tokens.
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TurboQuantError> {
        let text_c = CString::new(text)?;
        // Allocate a generous buffer — context size is the absolute max.
        let mut buf = vec![0i32; 32768];
        let n = unsafe {
            ffi::llamatq_tokenize(
                self.raw,
                text_c.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as i32,
                if add_special { 1 } else { 0 },
            )
        };
        if n < 0 {
            Err(TurboQuantError::EvalFailed(n))
        } else {
            buf.truncate(n as usize);
            Ok(buf)
        }
    }

    // ── API v3: Model metadata ────────────────────────────────────────────────

    /// Read a model metadata string value by key.
    /// Returns `None` if the key is not present in the model.
    pub fn model_meta(&self, key: &str) -> Option<String> {
        let key_c = CString::new(key).ok()?;
        let mut buf = vec![0u8; 8192];
        let n = unsafe {
            ffi::llamatq_model_meta(
                self.raw,
                key_c.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if n < 0 {
            None
        } else {
            let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
            Some(cstr.to_string_lossy().into_owned())
        }
    }

    // ── API v3: Chat evaluation ───────────────────────────────────────────────

    /// Evaluate a pre-formatted chat turn against the EXISTING KV cache.
    ///
    /// Unlike `eval_with_sampling`, this does NOT clear the KV cache, enabling
    /// multi-turn conversation. The `on_token` callback, if provided, is called
    /// once per generated token piece for real-time streaming into the TUI.
    ///
    /// Returns `(response_text, stats)` on success.
    ///
    /// ⚠️ Must be called from the same thread that created the context.
    pub fn chat_eval<F>(
        &self,
        formatted_turn: &str,
        max_tokens: i32,
        params: SamplingParams,
        mut on_token: Option<&mut F>,
    ) -> Result<(String, InferenceStats), TurboQuantError>
    where
        F: FnMut(&str),
    {
        let turn_c = CString::new(formatted_turn)?;
        let c_params: LlamaTqSamplingParams = params.into();
        let mut raw_stats = LlamaTqEvalStats::default();
        // 64 KB output buffer — enough for any reasonable response.
        let mut out_text = vec![0u8; 65536];

        // Build a fat-pointer trampoline so the C callback can call our Rust closure.
        // We use a *mut c_void pointing at the Option<&mut F> on this stack frame.
        struct CallbackState<'a, F: FnMut(&str)>(
            &'a mut Option<&'a mut F>,
        );

        // SAFETY: The C shim calls token_cb synchronously during chat_eval;
        //         it never stores the pointer or calls it after returning.
        unsafe extern "C" fn trampoline<F: FnMut(&str)>(
            piece: *const c_char,
            user_data: *mut c_void,
        ) {
            let cb_opt = &mut *(user_data as *mut Option<&mut F>);
            if let Some(cb) = cb_opt.as_mut() {
                if let Ok(s) = CStr::from_ptr(piece).to_str() {
                    cb(s);
                }
            }
        }

        let (cb_fn, user_data_ptr) = if on_token.is_some() {
            let ptr = &mut on_token as *mut Option<&mut F> as *mut c_void;
            (Some(trampoline::<F> as unsafe extern "C" fn(*const c_char, *mut c_void)), ptr)
        } else {
            (None, std::ptr::null_mut::<c_void>())
        };

        let result = unsafe {
            ffi::llamatq_chat_eval(
                self.raw,
                turn_c.as_ptr(),
                max_tokens,
                &c_params,
                &mut raw_stats,
                out_text.as_mut_ptr() as *mut c_char,
                out_text.len() as i32,
                cb_fn,
                user_data_ptr,
            )
        };

        if result < 0 {
            Err(TurboQuantError::EvalFailed(result))
        } else {
            // Trim the out_text buffer to the null terminator.
            let len = out_text.iter().position(|&b| b == 0).unwrap_or(out_text.len());
            out_text.truncate(len);
            let response = String::from_utf8_lossy(&out_text).into_owned();
            Ok((response, InferenceStats::from(raw_stats)))
        }
    }
}

impl Drop for TurboQuantCtx {
    fn drop(&mut self) {
        unsafe { ffi::llamatq_free(self.raw) }
    }
}
