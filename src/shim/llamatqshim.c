// src/shim/llamatqshim.c
#include "llamatqshim.h"
#include "llama.h"  // From llama-cpp-turboquant
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
    ctx_params.type_k = (enum ggml_type)params->cache_type_k;
    ctx_params.type_v = (enum ggml_type)params->cache_type_v;
    
    struct llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        return NULL;
    }
    
    struct llamatq_context* tq_ctx = 
        (struct llamatq_context*)malloc(sizeof(struct llamatq_context));
    if (!tq_ctx) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    
    tq_ctx->model = model;
    tq_ctx->ctx = ctx;
    tq_ctx->is_metal = 1; // Assuming Metal on Mac for this template
    
    return (llamatq_ctx)tq_ctx;
}

// FIX #6: Minimal inference loop pseudocode included
int llamatq_eval(llamatq_ctx ctx_ptr, const char* prompt, int max_tokens) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx || !prompt) return -1;
    
    // 1. Tokenize input
    int max_prompt_tokens = 2048;
    int32_t* tokens = (int32_t*)malloc(max_prompt_tokens * sizeof(int32_t));
    if (!tokens) return -1;
    
    // tokenize prompt
    int n_tokens = llama_tokenize(llama_model_get_vocab(ctx->model), prompt, strlen(prompt), tokens, max_prompt_tokens, true, true);
    if (n_tokens < 0) {
        free(tokens);
        return -1;
    }
    
    // 2. Process prompt tokens
    llama_memory_clear(llama_get_memory(ctx->ctx), false);
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(ctx->ctx, batch) != 0) {
        free(tokens);
        return -1;
    }
    
    // 3. Sampling loop
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    
    int n_past = n_tokens;
    int generated_tokens = 0;
    
    for (int i = 0; i < max_tokens; i++) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(smpl, ctx->ctx, -1);
        
        // Check for EOS
        if (llama_vocab_is_eog(llama_model_get_vocab(ctx->model), new_token)) {
            break;
        }
        
        // Print token
        char buf[128];
        int n_len = llama_token_to_piece(llama_model_get_vocab(ctx->model), new_token, buf, sizeof(buf), 0, true);
        if (n_len > 0) {
            printf("%.*s", n_len, buf);
            fflush(stdout);
        }
        
        // Decode new token
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx->ctx, batch) != 0) {
            break;
        }
        n_past++;
        generated_tokens++;
    }
    
    // 4. Cleanup and return
    llama_sampler_free(smpl);
    free(tokens);
    printf("\n");
    return generated_tokens;
}

void llamatq_free(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (ctx) {
        llama_free(ctx->ctx);
        llama_model_free(ctx->model);
        free(ctx);
    }
}

size_t llamatq_get_kv_cache_size(llamatq_ctx ctx_ptr) {
    return 0; // Return 0 since undocumented API was removed
}

int llamatq_is_metal_active(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return 0;
    return ctx->is_metal;
}
