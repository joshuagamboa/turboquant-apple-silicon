// src/template.rs — Chat template auto-detection and formatting
//
// Three template families supported:
//   - ChatML:          Qwen2/Qwen3, Yi, and others
//   - Llama3:          Llama-3.x models
//   - MistralInstruct: Mistral / Mixtral

use crate::context::TurboQuantCtx;

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful, accurate, concise AI assistant. \
     Follow the user's instructions, ask clarifying questions when needed, \
     and avoid making up facts.";

// ── Template enum ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// ChatML: used by Qwen2/3, Yi, and many community models.
    ChatML,
    /// Llama-3 / Llama-3.1 format.
    Llama3,
    /// Mistral / Mixtral instruct format.
    MistralInstruct,
}

impl std::fmt::Display for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatTemplate::ChatML => write!(f, "ChatML (Qwen/Yi)"),
            ChatTemplate::Llama3 => write!(f, "Llama-3"),
            ChatTemplate::MistralInstruct => write!(f, "Mistral Instruct"),
        }
    }
}

// ── Role enum ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System    => "system",
            Role::User      => "user",
            Role::Assistant => "assistant",
        }
    }
}

// ── Detection ─────────────────────────────────────────────────────────────────

impl ChatTemplate {
    /// Auto-detect the chat template from the model's GGUF metadata.
    ///
    /// Priority:
    /// 1. `tokenizer.chat_template` — pattern-match the Jinja template string
    /// 2. `general.architecture`    — match known architecture names
    /// 3. Fallback → `ChatML` (most universally compatible)
    pub fn detect(ctx: &TurboQuantCtx) -> Self {
        // 1. Try the explicit chat template string
        if let Some(tmpl) = ctx.model_meta("tokenizer.chat_template") {
            if let Some(kind) = Self::detect_from_template_str(&tmpl) {
                return kind;
            }
        }

        // 2. Fallback: check architecture name
        if let Some(arch) = ctx.model_meta("general.architecture") {
            let arch = arch.to_lowercase();
            if arch.contains("qwen") {
                return ChatTemplate::ChatML;
            }
            if arch.contains("llama") {
                return ChatTemplate::Llama3;
            }
            if arch.contains("mistral") || arch.contains("mixtral") {
                return ChatTemplate::MistralInstruct;
            }
        }

        // 3. Check general.name for further hints
        if let Some(name) = ctx.model_meta("general.name") {
            let name = name.to_lowercase();
            if name.contains("qwen") {
                return ChatTemplate::ChatML;
            }
            if name.contains("llama") {
                return ChatTemplate::Llama3;
            }
            if name.contains("mistral") || name.contains("mixtral") {
                return ChatTemplate::MistralInstruct;
            }
        }

        // 4. Safe default
        ChatTemplate::ChatML
    }

    /// Detect template type from the Jinja template string.
    fn detect_from_template_str(s: &str) -> Option<Self> {
        if s.contains("<|im_start|>") {
            Some(ChatTemplate::ChatML)
        } else if s.contains("<|start_header_id|>") {
            Some(ChatTemplate::Llama3)
        } else if s.contains("[INST]") {
            Some(ChatTemplate::MistralInstruct)
        } else {
            None
        }
    }
}

// ── Formatting ────────────────────────────────────────────────────────────────

impl ChatTemplate {
    /// Format a single message with role tags applied.
    ///
    /// For Mistral, system prompt is embedded inside the first [INST] block;
    /// this function does not handle that — use `format_system_message()`.
    pub fn format_message(&self, role: Role, content: &str) -> String {
        match self {
            ChatTemplate::ChatML => {
                format!("<|im_start|>{}\n{}<|im_end|>\n", role.as_str(), content)
            }
            ChatTemplate::Llama3 => {
                format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role.as_str(),
                    content
                )
            }
            ChatTemplate::MistralInstruct => {
                match role {
                    Role::System => {
                        // Mistral doesn't have a dedicated system role; embed as
                        // a <<SYS>> block inside the first user [INST].
                        // Returned as a prefix to be prepended to the first user message.
                        format!("<<SYS>>\n{}\n<</SYS>>\n\n", content)
                    }
                    Role::User => {
                        format!("[INST] {} [/INST]", content)
                    }
                    Role::Assistant => {
                        format!("{}</s>", content)
                    }
                }
            }
        }
    }

    /// The prefix to prepend immediately before generation begins (the
    /// "assistant turn header"). This primes the model to generate an assistant
    /// response.
    pub fn assistant_prefix(&self) -> &str {
        match self {
            ChatTemplate::ChatML => "<|im_start|>assistant\n",
            ChatTemplate::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ChatTemplate::MistralInstruct => " ",  // Generation follows immediately after [/INST]
        }
    }

    /// Return the default system prompt text.
    pub fn default_system_prompt() -> &'static str {
        DEFAULT_SYSTEM_PROMPT
    }

    /// Format a complete conversation turn ready for `chat_eval`.
    ///
    /// `user_content` is the raw user message text (without role tags).
    pub fn format_user_turn(&self, user_content: &str) -> String {
        let user_part = self.format_message(Role::User, user_content);
        format!("{}{}", user_part, self.assistant_prefix())
    }

    /// Format the initial system prompt message for the start of a session.
    pub fn format_session_prefix(&self, system_prompt: &str) -> String {
        match self {
            ChatTemplate::ChatML => {
                let sys = self.format_message(Role::System, system_prompt);
                // The first token in a ChatML conversation is the system message.
                sys
            }
            ChatTemplate::Llama3 => {
                // Llama-3 starts with a BOS token then the system header.
                let sys = self.format_message(Role::System, system_prompt);
                // BOS is inserted by the tokenizer (add_special=true in the
                // first llamatq_tokenize call).
                sys
            }
            ChatTemplate::MistralInstruct => {
                // Mistral/Mixtral has no dedicated system role.
                // The system prompt is embedded as <<SYS>>..</SYS> in the
                // first user [INST] block by the session layer.
                // Return an empty string here; the session prepends it to the
                // first user message.
                String::new()
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_from_template_string_chatml() {
        let s = "{% if message.role == 'system' %}<|im_start|>system\n...";
        assert_eq!(ChatTemplate::detect_from_template_str(s), Some(ChatTemplate::ChatML));
    }

    #[test]
    fn detect_from_template_string_llama3() {
        let s = "...{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>'...";
        assert_eq!(ChatTemplate::detect_from_template_str(s), Some(ChatTemplate::Llama3));
    }

    #[test]
    fn detect_from_template_string_mistral() {
        let s = "{{ '[INST] ' + messages[0].content + ' [/INST]' }}";
        assert_eq!(ChatTemplate::detect_from_template_str(s), Some(ChatTemplate::MistralInstruct));
    }

    #[test]
    fn detect_unknown_returns_none() {
        let s = "some random jinja template";
        assert_eq!(ChatTemplate::detect_from_template_str(s), None);
    }

    #[test]
    fn chatml_format_message() {
        let tmpl = ChatTemplate::ChatML;
        let msg = tmpl.format_message(Role::User, "Hello!");
        assert_eq!(msg, "<|im_start|>user\nHello!<|im_end|>\n");
    }

    #[test]
    fn llama3_format_message() {
        let tmpl = ChatTemplate::Llama3;
        let msg = tmpl.format_message(Role::Assistant, "Hi there");
        assert_eq!(msg, "<|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|>");
    }

    #[test]
    fn format_user_turn_ends_with_assistant_prefix() {
        let tmpl = ChatTemplate::ChatML;
        let turn = tmpl.format_user_turn("What is 2+2?");
        assert!(turn.ends_with("<|im_start|>assistant\n"));
    }
}
