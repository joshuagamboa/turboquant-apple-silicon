// src/context.rs
use crate::ffi::{self, LlamaTqCtx, LlamaTqParams, LlamaTqCacheType, NotSendSync};
use std::ffi::CString;
use std::marker::PhantomData;
use thiserror::Error;

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
    #[error("Evaluation not implemented - implement inference loop in llamatq_eval")]
    EvalNotImplemented,
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
    
    /// ⚠️ Must be called from the same thread that created the context
    pub fn eval(&self, prompt: &str, max_tokens: i32) -> Result<i32, TurboQuantError> {
        let prompt_c = CString::new(prompt)?;
        let result = unsafe { ffi::llamatq_eval(self.raw, prompt_c.as_ptr(), max_tokens) };
        
        if result < 0 {
            Err(TurboQuantError::EvalFailed(result))
        } else if result == 0 {
            Err(TurboQuantError::EvalNotImplemented)
        } else {
            Ok(result)
        }
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
