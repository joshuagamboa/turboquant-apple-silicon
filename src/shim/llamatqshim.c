// src/shim/llamatqshim.c
#include "llamatqshim.h"
#include "llama.h"  // From llama-cpp-turboquant
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mach/mach_time.h>  // Apple Silicon: mach_absolute_time()

// API version is passed via compiler flag from build.rs

// ── Timer helpers ────────────────────────────────────────────────────────────
// Returns current time in nanoseconds using mach_absolute_time.
// mach_absolute_time() is the highest-resolution monotonic clock on macOS/ARM.
static uint64_t now_ns(void) {
    return mach_absolute_time();
}

// Convert a mach_absolute_time delta to milliseconds.
// mach_timebase_info gives us the numer/denom to convert ticks → nanoseconds.
static double ticks_to_ms(uint64_t ticks) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double ns = (double)ticks * (double)info.numer / (double)info.denom;
    return ns / 1e6;
}

// ── Context struct ────────────────────────────────────────────────────────────
struct llamatq_context {
    struct llama_model* model;
    struct llama_context* ctx;
    struct llama_batch batch;
    int is_metal;
};

// ── API version ───────────────────────────────────────────────────────────────
int llamatq_get_api_version(void) {
    return LLAMA_TURBOQUANT_API_VERSION;
}

// ── Context creation ──────────────────────────────────────────────────────────
llamatq_ctx llamatq_create(const llamatq_params* params) {
    if (!params || !params->model_path) return NULL;

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params->n_gpu_layers;

    struct llama_model* model = llama_load_model_from_file(
        params->model_path, model_params);
    if (!model) return NULL;

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = params->n_ctx   > 0 ? params->n_ctx   : 2048;
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

    tq_ctx->model    = model;
    tq_ctx->ctx      = ctx;
    tq_ctx->is_metal = 1; // Apple Silicon: Metal always assumed active

    return (llamatq_ctx)tq_ctx;
}

// ── Core eval (with sampling + stats) ────────────────────────────────────────
int llamatq_eval_with_sampling(
    llamatq_ctx ctx_ptr,
    const char* prompt,
    int max_tokens,
    const llamatq_sampling_params* sparams,
    llamatq_eval_stats* out_stats)
{
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx || !prompt || !sparams) return -1;

    // Zero-initialise stats output early so partial results are always valid.
    if (out_stats) {
        out_stats->ttft_ms          = 0.0;
        out_stats->prompt_ms        = 0.0;
        out_stats->generation_ms    = 0.0;
        out_stats->total_ms         = 0.0;
        out_stats->prompt_tokens    = 0;
        out_stats->completion_tokens = 0;
        out_stats->prompt_tps       = 0.0;
        out_stats->generation_tps   = 0.0;
    }

    // ── t0: request start ────────────────────────────────────────────────────
    uint64_t t0 = now_ns();

    // 1. Tokenize prompt
    int max_prompt_tokens = llama_n_ctx(ctx->ctx);
    if (max_prompt_tokens <= 0) max_prompt_tokens = 2048;

    int32_t* tokens = (int32_t*)malloc(max_prompt_tokens * sizeof(int32_t));
    if (!tokens) return -1;

    int n_tokens = llama_tokenize(
        llama_model_get_vocab(ctx->model),
        prompt, (int)strlen(prompt),
        tokens, max_prompt_tokens,
        /*add_special=*/true,
        /*parse_special=*/true);
    if (n_tokens < 0) {
        free(tokens);
        return -2; // tokenization error
    }

    // 2. Decode the prompt batch
    llama_memory_clear(llama_get_memory(ctx->ctx), false);
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(ctx->ctx, batch) != 0) {
        free(tokens);
        return -3; // prompt decode error
    }

    // ── t1: prompt processing complete ───────────────────────────────────────
    uint64_t t1 = now_ns();

    // 3. Build sampler chain
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler* smpl = llama_sampler_chain_init(chain_params);

    if (sparams->temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(sparams->top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(sparams->temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(sparams->seed));
    }

    // 4. Generation loop
    int generated_tokens = 0;
    uint64_t t_first_token = 0; // set after first token is decoded

    for (int i = 0; i < max_tokens; i++) {
        if (n_tokens + generated_tokens >= max_prompt_tokens) {
            fprintf(stderr,
                "\n[Warning: Context size limit of %d tokens reached. Generation stopped early.]\n",
                max_prompt_tokens);
            break;
        }

        // Sample next token
        llama_token new_token = llama_sampler_sample(smpl, ctx->ctx, -1);

        // Check for end-of-generation
        if (llama_vocab_is_eog(llama_model_get_vocab(ctx->model), new_token)) {
            break;
        }

        // Print token immediately (streaming UX)
        char buf[128];
        int n_len = llama_token_to_piece(
            llama_model_get_vocab(ctx->model),
            new_token, buf, sizeof(buf), 0, true);
        if (n_len > 0) {
            printf("%.*s", n_len, buf);
            fflush(stdout);
        }

        // Decode the new token into the KV cache
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx->ctx, batch) != 0) {
            break;
        }

        generated_tokens++;

        // ── t_first: capture after first token is fully decoded ───────────────
        if (generated_tokens == 1) {
            t_first_token = now_ns();
        }
    }

    // ── t3: generation complete ───────────────────────────────────────────────
    uint64_t t3 = now_ns();

    llama_sampler_free(smpl);
    free(tokens);
    printf("\n");

    // 5. Populate stats
    if (out_stats) {
        double prompt_ms     = ticks_to_ms(t1 - t0);
        double generation_ms = ticks_to_ms(t3 - t1);
        double total_ms      = ticks_to_ms(t3 - t0);
        // TTFT: from request start to end of first token decode.
        // Falls back to prompt_ms if no token was generated.
        double ttft_ms = (t_first_token > 0)
            ? ticks_to_ms(t_first_token - t0)
            : prompt_ms;

        out_stats->ttft_ms           = ttft_ms;
        out_stats->prompt_ms         = prompt_ms;
        out_stats->generation_ms     = generation_ms;
        out_stats->total_ms          = total_ms;
        out_stats->prompt_tokens     = n_tokens;
        out_stats->completion_tokens = generated_tokens;
        out_stats->prompt_tps        = (prompt_ms     > 0.0) ? (n_tokens        / (prompt_ms     / 1000.0)) : 0.0;
        out_stats->generation_tps    = (generation_ms > 0.0) ? (generated_tokens / (generation_ms / 1000.0)) : 0.0;
    }

    return generated_tokens;
}

// ── Greedy convenience wrapper ────────────────────────────────────────────────
int llamatq_eval(llamatq_ctx ctx_ptr, const char* prompt, int max_tokens) {
    llamatq_sampling_params default_sparams = {
        .temperature = 0.0f,
        .top_p       = 1.0f,
        .seed        = 0,
    };
    return llamatq_eval_with_sampling(ctx_ptr, prompt, max_tokens, &default_sparams, NULL);
}

// ── Context teardown ──────────────────────────────────────────────────────────
void llamatq_free(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (ctx) {
        llama_free(ctx->ctx);
        llama_model_free(ctx->model);
        free(ctx);
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────────────
size_t llamatq_get_kv_cache_size(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return 0;
    return llama_state_get_size(ctx->ctx);
}

void llamatq_print_memory_breakdown(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (ctx) {
        llama_memory_breakdown_print(ctx->ctx);
    }
}

int llamatq_is_metal_active(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return 0;
    return ctx->is_metal;
}
