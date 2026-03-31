// include/llamatqshim.h  — API v3
#ifndef LLAMA_TQ_SHIM_H
#define LLAMA_TQ_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// API version is passed via compiler flag (-D), not defined here
// The shim expects LLAMA_TURBOQUANT_API_VERSION to be defined by build.rs

/**
 * Disable all internal llama.cpp logging. Useful for TUI applications
 * where standard error logs will corrupt the terminal screen.
 */
void llamatq_disable_logs(void);

/**
 * Initialize file logging. Redirects all llama.cpp and ggml logs
 * to the specified file path. Passing NULL disables logging.
 */
void llamatq_init_logging(const char* log_path);

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
 * Inference statistics collected during a single eval call.
 * All times are in milliseconds.
 * Pass a pointer to this struct as the last arg of llamatq_eval_with_sampling;
 * pass NULL to skip collection.
 */
typedef struct {
    double ttft_ms;           // Time to First Token: request start → first generated token (ms)
    double prompt_ms;         // Prompt processing time (ms)
    double generation_ms;     // Token generation time, excluding prompt (ms)
    double total_ms;          // Total wall-clock latency: request start → last token (ms)
    int    prompt_tokens;     // Number of input (prompt) tokens
    int    completion_tokens; // Number of output (generated) tokens
    double prompt_tps;        // Prompt processing throughput (tokens/sec)
    double generation_tps;    // Generation throughput (tokens/sec)
} llamatq_eval_stats;

/**
 * Create context with TurboQuant KV cache
 * @param params Context parameters
 * @return Context pointer or NULL on failure
 */
llamatq_ctx llamatq_create(const llamatq_params* params);

typedef struct {
    float temperature;   // 0.0 = greedy, > 0 = stochastic
    float top_p;         // 1.0 = disabled, < 1.0 = nucleus sampling
    uint32_t seed;       // RNG seed for reproducibility
} llamatq_sampling_params;

/**
 * Evaluate prompt using specific sampling parameters.
 * Timing statistics are written to out_stats if non-NULL.
 *
 * @param ctx       Context from llamatq_create
 * @param prompt    Input text
 * @param max_tokens Maximum tokens to generate
 * @param sparams   Sampling parameters (temperature, top_p, seed)
 * @param out_stats Output stats struct (nullable — pass NULL to skip)
 * @return Number of tokens generated, or negative value on error:
 *         -1  general failure
 *         -2  tokenization error
 *         -3  prompt decode error
 */
int llamatq_eval_with_sampling(
    llamatq_ctx ctx,
    const char* prompt,
    int max_tokens,
    const llamatq_sampling_params* sparams,
    llamatq_eval_stats* out_stats
);

/**
 * Evaluate prompt and return token count (using default greedy sampling).
 * Convenience wrapper; stats are not collected.
 * @param ctx Context from llamatq_create
 * @param prompt Input text
 * @param max_tokens Maximum tokens to generate
 * @return Number of tokens generated, or negative on error
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
 * Print detailed memory breakdown for the context to standard output/log
 */
void llamatq_print_memory_breakdown(llamatq_ctx ctx);

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

// ─── v3 API additions ─────────────────────────────────────────────────────────

/**
 * Token callback: called once per generated token with the token text.
 * @param token_text  Null-terminated UTF-8 string for this token piece.
 * @param user_data   Opaque pointer passed through from the caller.
 */
typedef void (*llamatq_token_callback)(const char* token_text, void* user_data);

/**
 * Clear the KV cache (all sequences).
 */
void llamatq_kv_clear(llamatq_ctx ctx);

/**
 * Get the number of tokens currently occupying the KV cache.
 * @return Token count, or 0 on error.
 */
int llamatq_kv_used(llamatq_ctx ctx);

/**
 * Remove KV positions [p0, p1) from sequence 0 and shift remaining
 * positions down by (p1 - p0). Used for sliding-window context management.
 * @param p0  Start of range to remove (inclusive).
 * @param p1  End of range to remove (exclusive). -1 = end of sequence.
 */
void llamatq_kv_shift(llamatq_ctx ctx, int p0, int p1);

/**
 * Tokenize text without decoding into the KV cache.
 * @param text        Input text.
 * @param out_tokens  Output buffer for token IDs.
 * @param max_tokens  Capacity of out_tokens.
 * @param add_special If non-zero, add BOS/EOS special tokens.
 * @return Number of tokens written, or negative on error.
 */
int llamatq_tokenize(
    llamatq_ctx ctx,
    const char* text,
    int32_t* out_tokens,
    int max_tokens,
    int add_special
);

/**
 * Read a model metadata string value by key.
 * @param key       Metadata key (e.g. "tokenizer.chat_template").
 * @param buf       Output buffer.
 * @param buf_size  Size of buf.
 * @return Number of bytes written (excluding null), or negative if not found.
 */
int llamatq_model_meta(
    llamatq_ctx ctx,
    const char* key,
    char* buf,
    int buf_size
);

/**
 * Chat evaluation: tokenise + decode into the EXISTING KV cache (no clear),
 * then generate until EOS or max_tokens. Unlike llamatq_eval_with_sampling,
 * this function does NOT reset the KV cache, enabling multi-turn conversations.
 *
 * Generated text is accumulated in out_text. If token_cb is non-NULL it is
 * called for each token as it is produced (real-time streaming).
 *
 * @param ctx            Context from llamatq_create.
 * @param formatted_turn Pre-formatted chat turn text (role tags + content).
 * @param max_tokens     Maximum tokens to generate.
 * @param sparams        Sampling parameters.
 * @param out_stats      Output stats (nullable).
 * @param out_text       Output buffer for generated text (nullable).
 * @param out_text_size  Size of out_text buffer.
 * @param token_cb       Per-token callback (nullable — disables streaming).
 * @param user_data      Opaque pointer forwarded to token_cb.
 * @return Number of tokens generated, or negative on error:
 *         -1  general failure
 *         -2  tokenisation error
 *         -3  prompt decode error
 */
int llamatq_chat_eval(
    llamatq_ctx ctx,
    const char* formatted_turn,
    int max_tokens,
    const llamatq_sampling_params* sparams,
    llamatq_eval_stats* out_stats,
    char* out_text,
    int out_text_size,
    llamatq_token_callback token_cb,
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TQ_SHIM_H
