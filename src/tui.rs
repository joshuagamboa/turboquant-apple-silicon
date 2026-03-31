// src/tui.rs — TurboQuant chat TUI built on tuinix
//
// Layout (rows, top→bottom):
//
//  ┌─────────────────────────────────────────────────────┐
//  │  TurboQuant Chat  [model]  [template]               │  ← header
//  ├─────────────────────────────────────────────────────┤
//  │                                                     │
//  │  History scroll area (user + assistant messages)    │
//  │                                                     │
//  │  [generating...]                                    │  ← streaming line
//  ├─────────────────────────────────────────────────────┤
//  │  > _                                                │  ← input line
//  ├─────────────────────────────────────────────────────┤
//  │  [ctx tokens]  [tps]  Ctrl+C exit  /reset /help    │  ← status bar
//  └─────────────────────────────────────────────────────┘

use std::fmt::{self, Write as FmtWrite};
use std::io::{self, Write};
use std::time::Duration;

use tuinix::{
    KeyCode, MouseEvent, MouseInput, Terminal, TerminalColor, TerminalEvent, TerminalFrame,
    TerminalInput, TerminalSize, TerminalStyle,
};

use crate::chat::ChatSession;
use crate::context::{InferenceStats, SamplingParams, TurboQuantCtx};

// ── Colour / style palette ────────────────────────────────────────────────────

fn style_header() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(0, 255, 150)) // vibrant teal/green
}
fn style_user() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(80, 200, 255)) // bright azure
}
fn style_assistant() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(245, 245, 245)) // pure white
}
fn style_streaming() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(255, 255, 150)) // soft yellow
}
fn style_status() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(150, 150, 170)) // muted blue-gray
}
fn style_error() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(255, 100, 100)) // vibrant red
}
fn style_input_prompt() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(0, 255, 128)) // spring green
}
fn style_input_text() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::WHITE)
}
fn style_dim() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(110, 110, 130))
}
fn style_think() -> TerminalStyle {
    TerminalStyle::new().italic().fg_color(TerminalColor::new(140, 140, 160)) // dim gray italic
}

// ── Message display record ────────────────────────────────────────────────────

struct DisplayMessage {
    is_user: bool,
    text:    String,
}

// ── ChatTui ───────────────────────────────────────────────────────────────────

pub struct ChatTui<'a> {
    terminal:    Terminal,
    session:     &'a mut ChatSession,
    ctx:         &'a TurboQuantCtx,
    params:      SamplingParams,
    max_tokens:  i32,
    model_name:  String,

    /// Rendered message history.
    messages:    Vec<DisplayMessage>,

    /// Line scroll offset for the history pane.
    scroll:      usize,

    /// Current user input buffer.
    input:       String,
    /// Cursor position within `input` (char index).
    cursor:      usize,

    /// Token being streamed currently.
    streaming_buf: String,
    is_generating: bool,

    /// last inference stats (displayed in status bar).
    last_stats:  Option<InferenceStats>,

    /// One-line notification (error / info) shown in the status bar.
    notification: Option<String>,
    notification_error: bool,

    /// Whether to lock the view to the bottom (auto-scroll).
    stick_to_bottom: bool,
    }


impl<'a> ChatTui<'a> {
    pub fn new(
        session:    &'a mut ChatSession,
        ctx:        &'a TurboQuantCtx,
        params:     SamplingParams,
        max_tokens: i32,
        model_name: impl Into<String>,
    ) -> io::Result<Self> {
        Ok(Self {
            terminal: Terminal::new()?,
            session,
            ctx,
            params,
            max_tokens,
            model_name: model_name.into(),
            messages:   Vec::new(),
            scroll:     0,
            input:      String::new(),
            cursor:     0,
            streaming_buf: String::new(),
            is_generating: false,
            last_stats:  None,
            notification: None,
            notification_error: false,
            stick_to_bottom: true,
        })
    }

    // ── Run loop ──────────────────────────────────────────────────────────────

    pub fn run(mut self) -> io::Result<()> {
        // Disable line-wrap to prevent scrolling out of bounds on full-pane draw
        print!("\x1b[?7l");
        // Enable mouse reporting: Button Motion (1002) + SGR Extended Mode (1006)
        // These are more robust for trackpads than the basic 1000 mode.
        print!("\x1b[?1002h\x1b[?1006h");
        let _ = io::stdout().flush();

        // Enable mouse capture via tuinix as well
        let _ = self.terminal.enable_mouse_input();

        // Initial welcome render
        self.redraw()?;

        loop {
            // poll_event with 50 ms timeout so we can react to resize etc.
            match self.terminal.poll_event(&[], &[], Some(Duration::from_millis(50)))? {
                Some(TerminalEvent::Input(input)) => {
                    match input {
                        TerminalInput::Key(key) => {
                            if !self.handle_key(key)? {
                                break; // Ctrl+C / escape
                            }
                        }
                        TerminalInput::Mouse(mouse) => {
                            self.handle_mouse(mouse)?;
                        }
                    }
                }
                Some(TerminalEvent::Resize(_)) => {
                    self.redraw()?;
                }
                Some(TerminalEvent::FdReady { .. }) => {}
                None => {} // timeout, no-op
            }
        }

        // Re-enable line wrap and disable mouse capture
        let _ = self.terminal.disable_mouse_input();
        print!("\x1b[?1002l\x1b[?1006l");
        print!("\x1b[?7h");
        let _ = io::stdout().flush();

        Ok(())
    }

    // ── Key handling ──────────────────────────────────────────────────────────

    /// Returns `false` to signal the run loop should exit.
    fn handle_key(&mut self, key: tuinix::KeyInput) -> io::Result<bool> {
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("turboquant.log") {
            use std::io::Write;
            let _ = writeln!(f, "[TUI] handle_key: code={:?} ctrl={} alt={}", key.code, key.ctrl, key.alt);
        }

        match key.code {
            // Ctrl+C / Ctrl+D → exit
            KeyCode::Char('c') | KeyCode::Char('d') if key.ctrl => {
                return Ok(false);
            }

            // Ctrl+U / Ctrl+D / Ctrl+B / Ctrl+F for scrolling (MacBook friendly)
            KeyCode::Char('u') | KeyCode::Char('b') if key.ctrl => {
                self.scroll = self.scroll.saturating_sub(10);
                self.stick_to_bottom = false;
                self.redraw()?;
            }
            KeyCode::Char('d') | KeyCode::Char('f') if key.ctrl => {
                self.scroll += 10;
                self.stick_to_bottom = false;
                self.redraw()?;
            }

            // Enter → submit
            KeyCode::Enter => {
                self.submit()?;
            }

            // Backspace
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let byte_idx = char_to_byte_idx(&self.input, self.cursor - 1);
                    let end_byte = char_to_byte_idx(&self.input, self.cursor);
                    self.input.drain(byte_idx..end_byte);
                    self.cursor -= 1;
                    self.redraw()?;
                }
            }

            // Left / Right cursor movement
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.redraw()?;
                }
            }
            KeyCode::Right => {
                let len = self.input.chars().count();
                if self.cursor < len {
                    self.cursor += 1;
                    self.redraw()?;
                }
            }

            // Home / End
            KeyCode::Home => {
                self.cursor = 0;
                self.redraw()?;
            }
            KeyCode::End => {
                self.cursor = self.input.chars().count();
                self.redraw()?;
            }

            // Scroll history up/down with page keys
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(5);
                self.stick_to_bottom = false;
                self.redraw()?;
            }
            KeyCode::PageDown => {
                self.scroll += 5;
                self.stick_to_bottom = false;
                self.redraw()?;
            }

            // Regular character input
            KeyCode::Char(ch) => {
                let byte_idx = char_to_byte_idx(&self.input, self.cursor);
                self.input.insert(byte_idx, ch);
                self.cursor += 1;
                self.notification = None;
                self.redraw()?;
            }

            // Delete key
            KeyCode::Delete => {
                let len = self.input.chars().count();
                if self.cursor < len {
                    let byte_idx = char_to_byte_idx(&self.input, self.cursor);
                    let end_byte = char_to_byte_idx(&self.input, self.cursor + 1);
                    self.input.drain(byte_idx..end_byte);
                    self.redraw()?;
                }
            }

            _ => {}
        }
        Ok(true)
    }

    fn handle_mouse(&mut self, mouse: MouseInput) -> io::Result<()> {
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("turboquant.log") {
            use std::io::Write;
            let _ = writeln!(f, "[TUI] handle_mouse: event={:?} pos={:?}", mouse.event, mouse.position);
        }

        match mouse.event {
            MouseEvent::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(3);
                self.stick_to_bottom = false;
                self.redraw()?;
            }
            MouseEvent::ScrollDown => {
                self.scroll += 3;
                // Note: we don't set stick_to_bottom here because we don't know max_scroll yet
                // The render() function will handle clamping.
                self.redraw()?;
            }
            _ => {}
        }
        Ok(())
    }

    // ── Submit user message ───────────────────────────────────────────────────

    fn submit(&mut self) -> io::Result<()> {
        let raw = self.input.trim().to_string();
        
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("turboquant.log") {
            use std::io::Write;
            let _ = writeln!(f, "[TUI] submit: raw_len={} input_len={}", raw.len(), self.input.len());
        }

        if raw.is_empty() {
            return Ok(());
        }

        // Handle slash commands
        match raw.as_str() {
            "/reset" => {
                self.session.reset(self.ctx);
                self.messages.clear();
                self.scroll = 0;
                self.streaming_buf.clear();
                self.last_stats = None;
                self.input.clear();
                self.cursor = 0;
                self.set_notification("Session reset.", false);
                self.redraw()?;
                return Ok(());
            }
            "/help" => {
                self.set_notification(
                    "Commands: /reset (clear history)  |  Ctrl+C to quit  |  PgUp/PgDn to scroll",
                    false,
                );
                self.input.clear();
                self.cursor = 0;
                self.redraw()?;
                return Ok(());
            }
            "/stats" => {
                if let Some(s) = &self.last_stats {
                    self.set_notification(
                        &format!(
                            "Last: TTFT {:.0}ms  gen {:.1} tok/s  {}/{} tokens",
                            s.ttft_ms, s.generation_tps, s.prompt_tokens, s.completion_tokens
                        ),
                        false,
                    );
                } else {
                    self.set_notification("No inference run yet.", false);
                }
                self.input.clear();
                self.cursor = 0;
                self.redraw()?;
                return Ok(());
            }
            _ => {}
        }

        // Record user message in display history
        self.messages.push(DisplayMessage { is_user: true, text: raw.clone() });
        self.input.clear();
        self.cursor = 0;
        self.is_generating = true;
        self.streaming_buf.clear();
        self.scroll_to_bottom();
        self.redraw()?;

        // ── Run inference (blocks this thread; token_cb redraws each token) ──

        // We need a mutable reference to self inside the callback.
        // Use a raw pointer trampoline: safe because inference runs synchronously.
        let params  = self.params;
        let max_tok = self.max_tokens;

        // Capture a *mut pointer to self — SAFE: inference blocks this thread.
        let self_ptr = self as *mut ChatTui<'a>;

        let mut on_token = |piece: &str| {
            let me = unsafe { &mut *self_ptr };
            me.streaming_buf.push_str(piece);
            // Redraw on every token (differential update keeps this fast).
            let _ = me.redraw();
        };

        let result = self.session.send(
            self.ctx,
            &raw,
            max_tok,
            params,
            Some(&mut on_token),
        );

        self.is_generating = false;

        match result {
            Ok((response, stats)) => {
                // Commit the streamed response as a display message.
                self.messages.push(DisplayMessage { is_user: false, text: response });
                self.streaming_buf.clear();
                self.last_stats = Some(stats);
                self.scroll_to_bottom();
            }
            Err(e) => {
                self.streaming_buf.clear();
                self.set_notification(&format!("Error: {}", e), true);
            }
        }

        self.redraw()?;

        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("turboquant.log") {
            use std::io::Write;
            let _ = writeln!(f, "[TUI] submit: inference loop returned");
        }

        Ok(())
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    fn redraw(&mut self) -> io::Result<()> {
        let size = self.terminal.size();
        
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("turboquant.log") {
            use std::io::Write;
            let _ = writeln!(f, "[TUI] redraw: size={:?} input_len={} cursor={}", size, self.input.len(), self.cursor);
        }

        let mut frame = TerminalFrame::new(size);
        self.render(&mut frame, size).map_err(io::Error::other)?;
        self.terminal.draw(frame)?;
        Ok(())
    }

    fn render(&mut self, frame: &mut TerminalFrame, size: TerminalSize) -> fmt::Result {
        let cols = size.cols as usize;
        let rows = size.rows as usize;

        if rows < 6 || cols < 30 {
            write!(frame, "Terminal too small")?;
            return Ok(());
        }

        // Layout rows:
        // row 0: Header
        // rows 1..(rows-4): History
        // row (rows-3): Separator
        // row (rows-2): Input prompt
        // row (rows-1): Status bar
        let history_rows = rows.saturating_sub(4);

        // ── 1. Header ──────────────────────────────────────────────────────────
        let header_style = style_header();
        let header_text = format!(
            " 󱚣 TurboQuant Chat │ {} │ {} │ {}",
            truncate_str(&self.model_name, 25),
            self.session.template_name(),
            self.session.context_summary(),
        );
        write_line(frame, cols, &header_text, header_style, true)?;

        // ── 2. History Pane ──────────────────────────────────────────────────
        let mut display_lines: Vec<(TerminalStyle, String)> = Vec::new();

        for msg in &self.messages {
            let (prefix, style) = if msg.is_user {
                ("  You › ", style_user())
            } else {
                ("   AI › ", style_assistant())
            };
            
            let wrapped = wrap_text(&msg.text, cols.saturating_sub(prefix.len()));
            for (i, line) in wrapped.iter().enumerate() {
                let line_text = if i == 0 {
                    format!("{}{}", prefix, line)
                } else {
                    format!("{}{}", " ".repeat(prefix.len()), line)
                };
                
                // Detection for thinking blocks (very basic)
                let current_style = if !msg.is_user && (line.contains("<think>") || line.contains("</think>")) {
                    style_think()
                } else {
                    style
                };
                
                display_lines.push((current_style, line_text));
            }
            display_lines.push((style, String::new())); // blank separator
        }

        // Streaming buffer if active
        if self.is_generating || !self.streaming_buf.is_empty() {
            let prefix = "   AI › ";
            let style = style_streaming();
            let wrapped = wrap_text(
                &format!("{}_", self.streaming_buf),
                cols.saturating_sub(prefix.len()),
            );
            for (i, line) in wrapped.iter().enumerate() {
                let line_text = if i == 0 {
                    format!("{}{}", prefix, line)
                } else {
                    format!("{}{}", " ".repeat(prefix.len()), line)
                };
                display_lines.push((style, line_text));
            }
        }

        let total_lines = display_lines.len();
        let max_scroll = total_lines.saturating_sub(history_rows);
        
        // If we are generating or the user hasn't manually scrolled up, stick to bottom
        let effective_scroll = if self.stick_to_bottom || self.is_generating {
            max_scroll
        } else {
            self.scroll.min(max_scroll)
        };
        
        let skip = effective_scroll;

        for i in 0..history_rows {
            let idx = skip + i;
            if idx < total_lines {
                let (style, ref text) = display_lines[idx];
                write_line(frame, cols, text, style, true)?;
            } else {
                // Empty lines to fill history area
                write_line(frame, cols, "", style_dim(), true)?;
            }
        }

        // ── 3. Separator ────────────────────────────────────────────────────────
        let sep_style = style_dim();
        let sep_char = "━".repeat(cols.saturating_sub(4)); // Thicker separator
        let separator = format!("  {}  ", sep_char);
        write_line(frame, cols, &separator, sep_style, true)?;

        // ── 4. Input Line ──────────────────────────────────────────────────────
        if self.is_generating {
            let msg = "  󰚩 Thinking... (Ctrl+C to stop)";
            write_line(frame, cols, msg, style_dim(), true)?;
        } else {
            let prompt = " ❯ ";
            let prompt_style = style_input_prompt();
            
            let max_input_w = cols.saturating_sub(prompt.len() + 2);
            let input_chars: Vec<char> = self.input.chars().collect();
            let visible_start = if input_chars.len() > max_input_w {
                input_chars.len() - max_input_w
            } else {
                0
            };
            let visible_chars: Vec<char> = input_chars[visible_start..].iter().cloned().collect();
            let cursor_visible = self.cursor.saturating_sub(visible_start);

            // Start line with prompt
            write!(frame, "{}{}", prompt_style, prompt)?;
            
            // Draw input text with cursor
            let text_style = style_input_text();
            write!(frame, "{}", text_style)?;
            
            for (i, &ch) in visible_chars.iter().enumerate() {
                if i == cursor_visible {
                    write!(frame, "{}{}{}{}", text_style.reverse(), ch, TerminalStyle::RESET, text_style)?;
                } else {
                    write!(frame, "{}", ch)?;
                }
            }
            if cursor_visible >= visible_chars.len() {
                write!(frame, "{} {}", text_style.reverse(), TerminalStyle::RESET)?;
            }
            
            // Pad to end of line
            let visible_len = prompt.len() + visible_chars.len() + (if cursor_visible >= visible_chars.len() { 1 } else { 0 });
            let padding = cols.saturating_sub(visible_len);
            write!(frame, "{}{}", TerminalStyle::RESET, " ".repeat(padding))?;
            writeln!(frame)?;
        }

        // ── 5. Status Bar ──────────────────────────────────────────────────────
        let stats_text = if let Some(s) = &self.last_stats {
            format!("{:.1} tok/s  TTFT {:.0}ms", s.generation_tps, s.ttft_ms)
        } else {
            "Ready".to_string()
        };

        let notif_text = if let Some(n) = &self.notification {
            n.clone()
        } else {
            "Ctrl+C: quit  |  /reset  |  /help  |  PgUp/Dn: scroll".to_string()
        };

        let status = format!(" {} │ {}", stats_text, notif_text);
        let status_style = if self.notification_error { style_error() } else { style_status() };
        
        // Final line: NO trailing newline
        write_line(frame, cols, &truncate_str(&status, cols), status_style, false)?;

        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn scroll_to_bottom(&mut self) {
        self.stick_to_bottom = true;
    }

    fn set_notification(&mut self, msg: &str, error: bool) {
        self.notification = Some(msg.to_string());
        self.notification_error = error;
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Write a fixed-width line padded to `cols`.
fn write_line(frame: &mut TerminalFrame, cols: usize, text: &str, style: TerminalStyle, newline: bool) -> fmt::Result {
    // Measure visible character count (ignore ANSI escape sequences).
    let visible_len = strip_ansi_len(text);
    let padding = cols.saturating_sub(visible_len);
    
    write!(
        frame,
        "{}{}{}{}",
        style,
        text,
        TerminalStyle::RESET,
        " ".repeat(padding)
    )?;
    
    if newline {
        writeln!(frame)?;
    }
    
    Ok(())
}

/// Wrap text to `max_width` characters per line.
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    for input_line in text.split('\n') {
        if input_line.is_empty() {
            lines.push(String::new());
            continue;
        }
        let words: Vec<&str> = input_line.split_whitespace().collect();
        let is_empty = words.is_empty();
        let mut current = String::new();
        for word in words {
            if current.is_empty() {
                current.push_str(word);
            } else if current.len() + 1 + word.len() <= max_width {
                current.push(' ');
                current.push_str(word);
            } else {
                lines.push(current.clone());
                current = word.to_string();
            }
        }
        if !current.is_empty() || is_empty {
            lines.push(current);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Truncate a string to at most `max` visible characters.
fn truncate_str(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        let mut t: String = chars[..max.saturating_sub(1)].iter().collect();
        t.push('…');
        t
    }
}

/// Convert a `char` index to a byte index within `s`.
fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

/// Estimate the display length of a string, ignoring ANSI escape sequences.
fn strip_ansi_len(s: &str) -> usize {
    let mut len = 0;
    let mut in_esc = false;
    for ch in s.chars() {
        if in_esc {
            if ch == 'm' { in_esc = false; }
        } else if ch == '\x1b' {
            in_esc = true;
        } else {
            len += 1;
        }
    }
    len
}
