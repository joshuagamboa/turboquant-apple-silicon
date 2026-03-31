// src/ffi.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use libc::{c_char, c_int, c_void, size_t};
use std::marker::PhantomData;

pub type LlamaTqCtx = *mut c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LlamaTqCacheType {
    F16    = 0,
    Q8_0   = 1,
    Turbo2 = 2,
    Turbo3 = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaTqParams {
    pub model_path:   *const c_char,
    pub n_gpu_layers: c_int,
    pub cache_type_k: LlamaTqCacheType,
    pub cache_type_v: LlamaTqCacheType,
    pub n_ctx:        u32,
    pub n_batch:      u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaTqSamplingParams {
    pub temperature: f32,
    pub top_p:       f32,
    pub seed:        u32,
}

/// Mirror of `llamatq_eval_stats` from the C shim.
/// All times are in milliseconds.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LlamaTqEvalStats {
    /// Time from request start to end of first token decode (ms).
    pub ttft_ms: f64,
    /// Prompt-processing time (ms).
    pub prompt_ms: f64,
    /// Token-generation time excluding prompt processing (ms).
    pub generation_ms: f64,
    /// Total wall-clock latency (ms).
    pub total_ms: f64,
    /// Number of input (prompt) tokens.
    pub prompt_tokens: c_int,
    /// Number of output (generated) tokens.
    pub completion_tokens: c_int,
    /// Prompt-processing throughput (tokens/sec).
    pub prompt_tps: f64,
    /// Generation throughput (tokens/sec).
    pub generation_tps: f64,
}

#[link(name = "llamatqshim")]
extern "C" {
    pub fn llamatq_create(params: *const LlamaTqParams) -> LlamaTqCtx;

    /// Convenience greedy-sampling wrapper; does not collect stats.
    pub fn llamatq_eval(ctx: LlamaTqCtx, prompt: *const c_char, max_tokens: c_int) -> c_int;

    /// Full eval: caller may pass a pointer to `LlamaTqEvalStats` to receive
    /// timing/usage data, or NULL to skip collection.
    pub fn llamatq_eval_with_sampling(
        ctx:      LlamaTqCtx,
        prompt:   *const c_char,
        max_tokens: c_int,
        sparams:  *const LlamaTqSamplingParams,
        out_stats: *mut LlamaTqEvalStats,
    ) -> c_int;

    pub fn llamatq_free(ctx: LlamaTqCtx);
    pub fn llamatq_get_kv_cache_size(ctx: LlamaTqCtx) -> size_t;
    pub fn llamatq_print_memory_breakdown(ctx: LlamaTqCtx);
    pub fn llamatq_is_metal_active(ctx: LlamaTqCtx) -> c_int;
    pub fn llamatq_get_api_version() -> c_int;
}

// Stable Rust !Send/!Sync marker (negative impls are unstable)
pub(crate) struct NotSendSync(PhantomData<*mut ()>);

impl NotSendSync {
    pub(crate) fn new() -> Self {
        Self(PhantomData)
    }
}
