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

// ── Logging toggle ────────────────────────────────────────────────────────────
static FILE* g_log_file = NULL;

static void dummy_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static void file_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)user_data;
    if (g_log_file) {
        fprintf(g_log_file, "%s", text);
        fflush(g_log_file);
    }
}

void llamatq_disable_logs(void) {
    llama_log_set(dummy_log_callback, NULL);
}

void llamatq_init_logging(const char* log_path) {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
    if (log_path) {
        g_log_file = fopen(log_path, "a");
        if (g_log_file) {
            fprintf(g_log_file, "\n\n=== TurboQuant Inference Session Started ===\n");
            llama_log_set(file_log_callback, NULL);
        }
    } else {
        llamatq_disable_logs();
    }
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

// \u2500\u2500 v3: KV cache management \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

void llamatq_kv_clear(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return;
    llama_memory_clear(llama_get_memory(ctx->ctx), false);
}

int llamatq_kv_used(llamatq_ctx ctx_ptr) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return 0;
    // seq_pos_max returns the largest position present in sequence 0.
    // Add 1 to convert from 0-indexed max position to token count.
    // Returns -1 if the sequence is empty.
    llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx->ctx), 0);
    return (pos < 0) ? 0 : (int)(pos + 1);
}

void llamatq_kv_shift(llamatq_ctx ctx_ptr, int p0, int p1) {
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx) return;
    llama_memory_t mem = llama_get_memory(ctx->ctx);
    // Remove the range [p0, p1) from sequence 0.
    llama_memory_seq_rm(mem, 0, (llama_pos)p0, (llama_pos)p1);
    // Shift remaining positions [p1, -1) down by (p1 - p0).
    llama_memory_seq_add(mem, 0, (llama_pos)p1, -1, -(llama_pos)(p1 - p0));
}

// \u2500\u2500 v3: Tokenization \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

int llamatq_tokenize(
    llamatq_ctx ctx_ptr,
    const char* text,
    int32_t* out_tokens,
    int max_tokens,
    int add_special)
{
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx || !text || !out_tokens) return -1;
    return llama_tokenize(
        llama_model_get_vocab(ctx->model),
        text, (int)strlen(text),
        out_tokens, max_tokens,
        add_special != 0,   // add_special (BOS)
        true);              // parse_special (needed for chat templates)
}

// \u2500\u2500 v3: Model metadata \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

int llamatq_model_meta(
    llamatq_ctx ctx_ptr,
    const char* key,
    char* buf,
    int buf_size)
{
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx || !key || !buf || buf_size <= 0) return -1;
    return llama_model_meta_val_str(ctx->model, key, buf, (size_t)buf_size);
}

// \u2500\u2500 v3: Chat evaluation (KV-cache continuations + token callback) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

int llamatq_chat_eval(
    llamatq_ctx ctx_ptr,
    const char* formatted_turn,
    int max_tokens,
    const llamatq_sampling_params* sparams,
    llamatq_eval_stats* out_stats,
    char* out_text,
    int out_text_size,
    llamatq_token_callback token_cb,
    void* user_data)
{
    struct llamatq_context* ctx = (struct llamatq_context*)ctx_ptr;
    if (!ctx || !formatted_turn || !sparams) return -1;

    // Zero-initialise stats early so partial results are always valid.
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

    // Prepare output text buffer
    int out_len = 0;
    if (out_text && out_text_size > 0) out_text[0] = '\0';

    uint64_t t0 = now_ns();

    // 1. Tokenize the formatted turn (do NOT add BOS; the template already has it)
    int n_ctx = llama_n_ctx(ctx->ctx);
    int32_t* tokens = (int32_t*)malloc(n_ctx * sizeof(int32_t));
    if (!tokens) return -1;

    int n_tokens = llama_tokenize(
        llama_model_get_vocab(ctx->model),
        formatted_turn, (int)strlen(formatted_turn),
        tokens, n_ctx,
        false,  // no BOS: the template supplies its own special tokens
        true);  // parse_special: needed to handle <|im_start|> etc.
    if (n_tokens < 0) {
        free(tokens);
        return -2;
    }

    // 2. Decode the turn into the EXISTING KV cache (no clear!)
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(ctx->ctx, batch) != 0) {
        free(tokens);
        return -3;
    }
    free(tokens);

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
    uint64_t t_first_token = 0;

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx->ctx, -1);

        // Check for end-of-generation
        if (llama_vocab_is_eog(llama_model_get_vocab(ctx->model), new_token)) {
            break;
        }

        // Decode token to text
        char piece[256];
        int n_piece = llama_token_to_piece(
            llama_model_get_vocab(ctx->model),
            new_token, piece, sizeof(piece), 0, true);
        if (n_piece > 0) {
            piece[n_piece] = '\0';

            // Fire the streaming callback (for TUI / caller streaming)
            if (token_cb) {
                token_cb(piece, user_data);
            }

            // Accumulate into out_text if provided
            if (out_text && out_len + n_piece < out_text_size - 1) {
                memcpy(out_text + out_len, piece, n_piece);
                out_len += n_piece;
                out_text[out_len] = '\0';
            }
        }

        // Decode token into KV cache
        struct llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx->ctx, next_batch) != 0) break;

        generated_tokens++;
        if (generated_tokens == 1) t_first_token = now_ns();

        // Guard: stop if KV cache is nearly full
        llama_pos cur_max = llama_memory_seq_pos_max(llama_get_memory(ctx->ctx), 0);
        if (cur_max >= n_ctx - 4) break;
    }

    uint64_t t3 = now_ns();
    llama_sampler_free(smpl);

    // 5. Populate stats
    if (out_stats) {
        double prompt_ms     = ticks_to_ms(t1 - t0);
        double generation_ms = ticks_to_ms(t3 - t1);
        double total_ms      = ticks_to_ms(t3 - t0);
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

