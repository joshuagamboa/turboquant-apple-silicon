// include/llamatqshim.h
#ifndef LLAMA_TQ_SHIM_H
#define LLAMA_TQ_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// API version is passed via compiler flag (-D), not defined here
// The shim expects LLAMA_TURBOQUANT_API_VERSION to be defined by build.rs

typedef void* llamatq_ctx;

// Cache type enumeration (fork-specific)
typedef enum {
    LLAMA_TQ_CACHE_F16 = 0,
    LLAMA_TQ_CACHE_Q8_0 = 1,
    LLAMA_TQ_CACHE_TURBO2 = 2,
    LLAMA_TQ_CACHE_TURBO3 = 3,
} llamatq_cache_type;

// Context parameters
typedef struct {
    const char* model_path;
    int n_gpu_layers;
    llamatq_cache_type cache_type_k;
    llamatq_cache_type cache_type_v;
    uint32_t n_ctx;
    uint32_t n_batch;
} llamatq_params;

/**
 * Create context with TurboQuant KV cache
 * @param params Context parameters
 * @return Context pointer or NULL on failure
 */
llamatq_ctx llamatq_create(const llamatq_params* params);

/**
 * Evaluate prompt and return token count
 * @param ctx Context from llamatq_create
 * @param prompt Input text
 * @param max_tokens Maximum tokens to generate
 * @return Number of tokens generated, or -1 on error
 */
int llamatq_eval(llamatq_ctx ctx, const char* prompt, int max_tokens);

/**
 * Free context and release resources
 */
void llamatq_free(llamatq_ctx ctx);

/**
 * Get KV cache size in bytes (for verification)
 */
size_t llamatq_get_kv_cache_size(llamatq_ctx ctx);

/**
 * Check if Metal backend is active
 * @return 1 if Metal active, 0 otherwise
 */
int llamatq_is_metal_active(llamatq_ctx ctx);

/**
 * Get API version (runtime check)
 * @return API version number
 */
int llamatq_get_api_version(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TQ_SHIM_H
