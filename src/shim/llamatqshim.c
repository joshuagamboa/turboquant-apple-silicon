// src/shim/llamatqshim.c
#include "llamatqshim.h"
#include "llama.h"  // From llama-cpp-turboquant
#include <stdlib.h>
#include <string.h>

// API version is passed via compiler flag from build.rs

struct llamatq_context {
    struct llama_model* model;
    struct llama_context* ctx;
    struct llama_batch batch;
    int is_metal;
};

int llamatq_get_api_version(void) {
    return LLAMA_TURBOQUANT_API_VERSION;
}

llamatq_ctx llamatq_create(const llamatq_params* params) {
    if (!params || !params->model_path) return NULL;
    
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params->n_gpu_layers;
    
    struct llama_model* model = llama_load_model_from_file(
        params->model_path, model_params);
    if (!model) return NULL;
    
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params->n_ctx > 0 ? params->n_ctx : 2048;
    ctx_params.n_batch = params->n_batch > 0 ? params->n_batch : 512;
    
    // TurboQuant-specific: set cache types (FORK API)
    ctx_params.type_k = llama_kv_cache_type_from_int(
        (int)params->cache_type_k);
    ctx_params.type_v = llama_kv_cache_type_from_int(
        (int)params->cache_type_v);
    
    struct llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        llama_free_model(model);
        return NULL;
    }
    
    struct llamatq_context* tq_ctx = 
        (struct llamatq_context*)malloc(sizeof(struct llamatq_context));
    if (!tq_ctx) {
        llama_free(ctx);
        llama_free_model(model);
        return NULL;
    }
    
    tq_ctx->model = model;
    tq_ctx->ctx = ctx;
    tq_ctx->batch = llama_batch_get_one(NULL, params->n_batch);
    tq_ctx->is_metal = llama_backend_is_metal();
    
    return (llamatq_ctx)tq_ctx;
}

// FIX #6: Minimal inference loop pseudocode included
int llamatq_eval(llamatq_ctx ctx_ptr, const char* prompt, int max_tokens) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx || !prompt) return -1;
    
    /*
     * ⚠️ SKELETON IMPLEMENTATION - YOU MUST IMPLEMENT THIS
     * 
     * Minimal inference loop structure:
     * 
     * // 1. Tokenize input
     * int32_t tokens[512];
     * int n_tokens = llama_tokenize(ctx->model, prompt, strlen(prompt), 
     *                               tokens, 512, true, true);
     * 
     * // 2. Process prompt tokens
     * struct llama_batch batch = llama_batch_get_one(tokens, n_tokens, 0, 0);
     * llama_decode(ctx->ctx, batch);
     * 
     * // 3. Sampling loop (repeat until max_tokens or EOS)
     * for (int i = 0; i < max_tokens; i++) {
     *     // Sample next token
     *     llama_token new_token = llama_sampling_sample(sampler, ctx->ctx, NULL, 0);
     *     
     *     // Check for EOS
     *     if (new_token == llama_token_eos(ctx->model)) break;
     *     
     *     // Decode new token
     *     batch = llama_batch_get_one(&new_token, 1, n_tokens + i, 0);
     *     llama_decode(ctx->ctx, batch);
     * }
     * 
     * // 4. Return total tokens generated
     * return i;
     * 
     * Reference implementation:
     * https://github.com/ggml-org/llama.cpp/blob/master/examples/main/main.cpp
     */
    
    // Placeholder - returns 0 to indicate not implemented
    return 0;
}

void llamatq_free(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (ctx) {
        llama_free(ctx->ctx);
        llama_free_model(ctx->model);
        free(ctx);
    }
}

size_t llamatq_get_kv_cache_size(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return 0;
    return llama_get_kv_cache_size(ctx->ctx);
}

int llamatq_is_metal_active(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return 0;
    return ctx->is_metal;
}
