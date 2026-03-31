// src/chat.rs — ChatSession: stateful multi-turn conversation management
//
// Owns:
//   - Message history (Vec<ChatMessage>)
//   - KV position tracking (n_past)
//   - Context windowing logic (KV Shift primary, Re-process fallback)
//   - Template-aware formatting

use crate::context::{InferenceStats, SamplingParams, TurboQuantCtx, TurboQuantError};
use crate::template::{ChatTemplate, Role};

// ── Data types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role:    Role,
    pub content: String,
    /// Token count for this message (set after the message is evaluated).
    pub n_tokens: i32,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), n_tokens: 0 }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), n_tokens: 0 }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), n_tokens: 0 }
    }
}

// ── Windowing configuration ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    /// Context size in tokens (must match the one passed to TurboQuantCtx::new).
    pub ctx_size: i32,
    /// Fraction of the context to keep when windowing triggers (0.0–1.0).
    /// Default: 0.75 — drop the oldest 25 % of the context.
    pub keep_ratio: f32,
    /// Reserve this many tokens for the assistant's upcoming response.
    pub response_reserve: i32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            ctx_size:         8192,
            keep_ratio:       0.75,
            response_reserve: 512,
        }
    }
}

impl WindowConfig {
    /// Maximum KV tokens we may consume before windowing must happen.
    pub fn high_water_mark(&self) -> i32 {
        self.ctx_size - self.response_reserve
    }

    /// Target number of tokens to retain after windowing.
    pub fn target_tokens(&self) -> i32 {
        (self.ctx_size as f32 * self.keep_ratio) as i32
    }

    /// Number of tokens to evict when windowing triggers.
    pub fn evict_tokens(&self) -> i32 {
        self.ctx_size - self.target_tokens()
    }
}

// ── ChatSession ───────────────────────────────────────────────────────────────

pub struct ChatSession {
    pub messages:  Vec<ChatMessage>,
    pub template:  ChatTemplate,
    pub config:    WindowConfig,

    /// Number of tokens currently in the KV cache (tracked in-sync with C).
    n_past: i32,

    /// Whether the system prompt has been injected into the KV cache yet.
    system_injected: bool,
}

impl ChatSession {
    /// Create a new session with template auto-detection and default system prompt.
    pub fn new(ctx: &TurboQuantCtx, config: WindowConfig) -> Self {
        let template = ChatTemplate::detect(ctx);
        Self {
            messages: Vec::new(),
            template,
            config,
            n_past: 0,
            system_injected: false,
        }
    }

    /// Create a new session with an explicit template.
    pub fn with_template(template: ChatTemplate, config: WindowConfig) -> Self {
        Self {
            messages: Vec::new(),
            template,
            config,
            n_past: 0,
            system_injected: false,
        }
    }

    /// KV tokens consumed.
    pub fn n_past(&self) -> i32 { self.n_past }

    /// Human-readable template name.
    pub fn template_name(&self) -> String { self.template.to_string() }

    // ── Core send/receive ─────────────────────────────────────────────────────

    /// Send a user message, run inference, and return the assistant response.
    ///
    /// `on_token`: optional streaming callback; called once per token piece.
    ///
    /// Returns `(assistant_response, stats)`.
    pub fn send<F>(
        &mut self,
        ctx:        &TurboQuantCtx,
        user_input: &str,
        max_tokens: i32,
        params:     SamplingParams,
        on_token:   Option<&mut F>,
    ) -> Result<(String, InferenceStats), TurboQuantError>
    where
        F: FnMut(&str),
    {
        // 1. Inject system prompt on the first turn
        if !self.system_injected {
            self.inject_system(ctx, params)?;
        }

        // 2. Estimate new tokens and apply windowing if needed
        let user_formatted = self.template.format_user_turn(user_input);
        let user_tokens = ctx.tokenize(&user_formatted, false)?.len() as i32;

        if self.n_past + user_tokens >= self.config.high_water_mark() {
            self.apply_windowing(ctx)?;
        }

        // 3. Run chat_eval — appends to existing KV cache
        let (response, stats) = ctx.chat_eval(
            &user_formatted,
            max_tokens,
            params,
            on_token,
        )?;

        // 4. Track KV usage (use C's count as ground truth)
        self.n_past = ctx.kv_used();

        // 5. Record messages in history
        let mut user_msg = ChatMessage::user(user_input);
        user_msg.n_tokens = user_tokens;
        self.messages.push(user_msg);

        let completion_tokens = stats.completion_tokens;
        let mut asst_msg = ChatMessage::assistant(&response);
        asst_msg.n_tokens = completion_tokens;
        self.messages.push(asst_msg);

        Ok((response, stats))
    }

    // ── System prompt injection ───────────────────────────────────────────────

    fn inject_system(
        &mut self,
        ctx:    &TurboQuantCtx,
        params: SamplingParams,
    ) -> Result<(), TurboQuantError> {
        let system_text = ChatTemplate::default_system_prompt();
        let prefix = self.template.format_session_prefix(system_text);

        if !prefix.is_empty() {
            // Decode the system prefix into the KV cache.  We pass it as a "turn"
            // with no generation (max_tokens = 0) to avoid producing any output.
            // Use a dummy 0-max to just process without generating.
            let mut noop: Option<&mut dyn FnMut(&str)> = None;
            let _result = ctx.chat_eval(&prefix, 0, params, noop.as_mut());
            // Even if the result is 0 tokens generated, n_past should increase.
        }

        self.n_past = ctx.kv_used();
        self.system_injected = true;

        let mut sys_msg = ChatMessage::system(system_text);
        sys_msg.n_tokens = self.n_past;
        self.messages.push(sys_msg);

        Ok(())
    }

    // ── Windowing ─────────────────────────────────────────────────────────────

    /// Apply in-place KV shift (primary strategy).
    ///
    /// Evicts the oldest `evict_tokens` tokens from the KV cache, preserving
    /// the most recent context. If the shift fails, falls back to re-process.
    fn apply_windowing(
        &mut self,
        ctx: &TurboQuantCtx,
    ) -> Result<(), TurboQuantError> {
        // We want to evict from the middle, preserving the system prompt (pos 0..X)
        // and the most recent history.
        let sys_tokens = if !self.messages.is_empty() && self.messages[0].role == Role::System {
            self.messages[0].n_tokens.max(1)
        } else {
            0
        };

        let evict = self.config.evict_tokens();
        // Shift range [sys_tokens, sys_tokens + evict) out.
        ctx.kv_shift(sys_tokens, sys_tokens + evict);

        // Verify the shift succeeded by checking the new KV occupancy.
        let new_n_past = ctx.kv_used();
        if new_n_past >= self.n_past {
            // Shift didn't reduce usage — fall back to re-process.
            eprintln!(
                "[chat] KV shift failed (n_past unchanged: {}); falling back to re-process",
                new_n_past
            );
            self.apply_reprocess(ctx)?;
        } else {
            self.n_past = new_n_past;
            eprintln!(
                "[chat] Context window shifted: evicted {} tokens, n_past {} → {}",
                evict, self.n_past, new_n_past
            );
        }

        Ok(())
    }

    /// Re-process strategy fallback: clear KV cache and re-decode the recent
    /// message history from scratch.
    fn apply_reprocess(&mut self, ctx: &TurboQuantCtx) -> Result<(), TurboQuantError> {
        ctx.kv_clear();
        self.n_past = 0;
        self.system_injected = false;

        // Rebuild from the most recent messages that fit within target_tokens.
        let target = self.config.target_tokens();
        let mut rebuilt: Vec<ChatMessage> = Vec::new();
        let mut token_estimate = 0i32;

        // Walk history in reverse to collect messages that fit.
        for msg in self.messages.iter().rev() {
            let est = msg.n_tokens.max(32); // floor at 32 in case n_tokens wasn't set
            if token_estimate + est > target {
                break;
            }
            token_estimate += est;
            rebuilt.push(msg.clone());
        }
        rebuilt.reverse();

        // Archive old messages (replace history with only the kept window).
        self.messages = rebuilt;

        // Re-inject system prompt
        let system_text = ChatTemplate::default_system_prompt();
        let prefix = self.template.format_session_prefix(system_text);
        if !prefix.is_empty() {
            let params = SamplingParams::default();
            let mut noop: Option<&mut dyn FnMut(&str)> = None;
            let _r = ctx.chat_eval(&prefix, 0, params, noop.as_mut());
        }
        self.system_injected = true;

        // Re-decode the kept history (user + assistant turns).
        let params = SamplingParams::default();
        for msg in &self.messages {
            if msg.role == Role::System { continue; }
            let formatted = match msg.role {
                Role::User      => self.template.format_user_turn(&msg.content),
                Role::Assistant => self.template.format_message(Role::Assistant, &msg.content),
                Role::System    => continue,
            };
            let mut noop: Option<&mut dyn FnMut(&str)> = None;
            let _r = ctx.chat_eval(&formatted, 0, params, noop.as_mut());
        }

        self.n_past = ctx.kv_used();
        eprintln!(
            "[chat] Re-processed context: {} messages, n_past {}",
            self.messages.len(),
            self.n_past
        );

        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Return a summary string for display in the TUI status bar.
    pub fn context_summary(&self) -> String {
        format!(
            "{}/{} tokens ({:.0}%)",
            self.n_past,
            self.config.ctx_size,
            (self.n_past as f64 / self.config.ctx_size as f64) * 100.0,
        )
    }

    /// Reset the session: clear history, KV cache, and position counter.
    pub fn reset(&mut self, ctx: &TurboQuantCtx) {
        ctx.kv_clear();
        self.messages.clear();
        self.n_past = 0;
        self.system_injected = false;
    }
}
