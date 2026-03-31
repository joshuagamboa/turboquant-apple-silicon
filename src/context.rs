// src/context.rs
use crate::ffi::{
    self, LlamaTqCacheType, LlamaTqCtx, LlamaTqParams, LlamaTqSamplingParams, NotSendSync,
};
use std::ffi::CString;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

impl From<SamplingParams> for LlamaTqSamplingParams {
    fn from(params: SamplingParams) -> Self {
        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            seed: params.seed,
        }
    }
}

#[derive(Error, Debug)]
pub enum TurboQuantError {
    #[error("Failed to create context: null pointer returned")]
    CreateFailed,
    #[error("Failed to evaluate: {0}")]
    EvalFailed(i32),
    #[error("Metal backend not active")]
    MetalNotActive,
    #[error("Invalid UTF-8 in path or prompt")]
    InvalidUtf8(#[from] std::ffi::NulError),
    #[error("API version mismatch: expected {expected}, got {actual}")]
    ApiVersionMismatch { expected: i32, actual: i32 },
    #[error("Context is not thread-safe - do not send across threads")]
    NotThreadSafe,
}

pub struct TurboQuantCtx {
    raw: LlamaTqCtx,
    _marker: PhantomData<NotSendSync>,
}

impl TurboQuantCtx {
    const EXPECTED_API_VERSION: i32 = 1;
    
    /// Create a new TurboQuant context
    /// 
    /// ⚠️ **THREADING**: This context must be used from a single OS thread.
    /// Do not send across threads or use in async executors without pinning.
    pub fn new(
        model_path: &str,
        n_gpu_layers: i32,
        cache_type: LlamaTqCacheType,
        n_ctx: u32,
        n_batch: u32,
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
            model_path: model_path_c.as_ptr(),
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
        
        Ok(Self { 
            raw,
            _marker: PhantomData,
        })
    }
    
    pub fn is_metal_active(&self) -> bool {
        unsafe { ffi::llamatq_is_metal_active(self.raw) != 0 }
    }
    
    pub fn kv_cache_size(&self) -> usize {
        unsafe { ffi::llamatq_get_kv_cache_size(self.raw) }
    }
    
    /// Evaluate the prompt using specific sampling parameters
    /// ⚠️ Must be called from the same thread that created the context
    pub fn eval_with_sampling(
        &self,
        prompt: &str,
        max_tokens: i32,
        params: SamplingParams,
    ) -> Result<i32, TurboQuantError> {
        let prompt_c = CString::new(prompt)?;
        let c_params: LlamaTqSamplingParams = params.into();

        let result = unsafe {
            ffi::llamatq_eval_with_sampling(
                self.raw,
                prompt_c.as_ptr(),
                max_tokens,
                &c_params,
            )
        };

        if result < 0 {
            Err(TurboQuantError::EvalFailed(result))
        } else {
            Ok(result)
        }
    }

    /// Evaluate the prompt using default (greedy) sampling
    /// ⚠️ Must be called from the same thread that created the context
    pub fn eval(&self, prompt: &str, max_tokens: i32) -> Result<i32, TurboQuantError> {
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
