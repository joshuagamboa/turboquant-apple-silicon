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
use std::io;
use std::time::Duration;

use tuinix::{
    KeyCode, Terminal, TerminalColor, TerminalEvent, TerminalFrame, TerminalInput, TerminalSize,
    TerminalStyle,
};

use crate::chat::ChatSession;
use crate::context::{InferenceStats, SamplingParams, TurboQuantCtx};

// ── Colour / style palette ────────────────────────────────────────────────────

fn style_header() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(30, 215, 160)) // teal
}
fn style_user() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(100, 180, 255)) // light blue
}
fn style_assistant() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(230, 230, 230)) // near-white
}
fn style_streaming() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(200, 200, 100)) // warm yellow
}
fn style_status() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(140, 140, 160)) // muted
}
fn style_error() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::new(255, 80, 80)) // red
}
fn style_input() -> TerminalStyle {
    TerminalStyle::new().bold().fg_color(TerminalColor::WHITE)
}
fn style_dim() -> TerminalStyle {
    TerminalStyle::new().fg_color(TerminalColor::new(90, 90, 110))
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

    /// Last inference stats (displayed in status bar).
    last_stats:  Option<InferenceStats>,

    /// One-line notification (error / info) shown in the status bar.
    notification: Option<String>,
    notification_error: bool,
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
        })
    }

    // ── Run loop ──────────────────────────────────────────────────────────────

    pub fn run(mut self) -> io::Result<()> {
        // Initial welcome render
        self.redraw()?;

        loop {
            // poll_event with 50 ms timeout so we can react to resize etc.
            match self.terminal.poll_event(&[], &[], Some(Duration::from_millis(50)))? {
                Some(TerminalEvent::Input(input)) => {
                    let TerminalInput::Key(key) = input else { continue };
                    if !self.handle_key(key.code)? {
                        break; // Ctrl+C / escape
                    }
                }
                Some(TerminalEvent::Resize(_)) => {
                    self.redraw()?;
                }
                Some(TerminalEvent::FdReady { .. }) => {}
                None => {} // timeout, no-op
            }
        }

        Ok(())
    }

    // ── Key handling ──────────────────────────────────────────────────────────

    /// Returns `false` to signal the run loop should exit.
    fn handle_key(&mut self, code: KeyCode) -> io::Result<bool> {
        match code {
            // Ctrl+C / Ctrl+D → exit
            KeyCode::Char('c') | KeyCode::Char('d') => {
                // tuinix provides modifier keys through KeyInput.modifiers; since we
                // only have `code` here we check both Ctrl chars. In practice the
                // terminal sends ETX (0x03) for Ctrl+C which KeyCode maps to Char('c').
                // A cleaner check would use modifiers but this works for our use case.
                return Ok(false);
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
                self.redraw()?;
            }
            KeyCode::PageDown => {
                self.scroll += 5;
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

    // ── Submit user message ───────────────────────────────────────────────────

    fn submit(&mut self) -> io::Result<()> {
        let raw = self.input.trim().to_string();
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
        Ok(())
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    fn redraw(&mut self) -> io::Result<()> {
        let size = self.terminal.size();
        let mut frame = TerminalFrame::new(size);
        self.render(&mut frame, size).map_err(io::Error::other)?;
        self.terminal.draw(frame)?;
        Ok(())
    }

    fn render(&self, frame: &mut TerminalFrame, size: TerminalSize) -> fmt::Result {
        let cols = size.cols as usize;
        let rows = size.rows as usize;

        // Safety: rows can be very small in theory.
        if rows < 5 || cols < 20 {
            write!(frame, "Terminal too small")?;
            return Ok(());
        }

        // Row budget:
        //   row 0          → header
        //   rows 1..(n-3)  → history + streaming
        //   row n-3        → separator
        //   row n-2        → input line
        //   row n-1        → status bar
        let history_rows = rows.saturating_sub(4);

        // ── Header ────────────────────────────────────────────────────────────
        let header_text = format!(
            " TurboQuant Chat │ {} │ {} │ {}",
            truncate_str(&self.model_name, 30),
            self.session.template_name(),
            self.session.context_summary(),
        );
        write_line(frame, cols, &header_text, style_header())?;

        // ── History pane ──────────────────────────────────────────────────────
        // Collect all lines to display
        let mut display_lines: Vec<(bool, String)> = Vec::new(); // (is_user, line)

        for msg in &self.messages {
            let prefix = if msg.is_user { "  You › " } else { "   AI › " };
            let wrapped = wrap_text(&msg.text, cols.saturating_sub(prefix.len()));
            for (i, line) in wrapped.iter().enumerate() {
                if i == 0 {
                    display_lines.push((msg.is_user, format!("{}{}", prefix, line)));
                } else {
                    let pad = " ".repeat(prefix.len());
                    display_lines.push((msg.is_user, format!("{}{}", pad, line)));
                }
            }
            display_lines.push((msg.is_user, String::new())); // blank line separator
        }

        // If generating, show the streaming buffer with a cursor
        let streaming_lines = if self.is_generating || !self.streaming_buf.is_empty() {
            let prefix = "   AI › ";
            let wrapped = wrap_text(
                &format!("{}_", self.streaming_buf),
                cols.saturating_sub(prefix.len()),
            );
            let mut sl: Vec<String> = Vec::new();
            for (i, line) in wrapped.iter().enumerate() {
                if i == 0 {
                    sl.push(format!("{}{}", prefix, line));
                } else {
                    let pad = " ".repeat(prefix.len());
                    sl.push(format!("{}{}", pad, line));
                }
            }
            sl
        } else {
            Vec::new()
        };

        // Join all lines and apply scroll
        let total_lines = display_lines.len() + streaming_lines.len();
        let max_scroll = total_lines.saturating_sub(history_rows);
        let effective_scroll = self.scroll.min(max_scroll);

        let all_lines_count = display_lines.len() + streaming_lines.len();
        let skip = if all_lines_count > history_rows {
            // Auto-scroll to bottom unless user has scrolled up.
            if effective_scroll == 0 {
                all_lines_count - history_rows
            } else {
                effective_scroll
            }
        } else {
            0
        };

        // Draw history_rows lines
        let mut rendered_history = 0usize;
        let combined_len = display_lines.len() + streaming_lines.len();

        for row_idx in 0..history_rows {
            let line_idx = skip + row_idx;
            if line_idx >= combined_len {
                // Blank padding
                write_line(frame, cols, "", style_dim())?;
                continue;
            }

            if line_idx < display_lines.len() {
                let (is_user, ref line) = display_lines[line_idx];
                let style = if is_user { style_user() } else { style_assistant() };
                write_line(frame, cols, line, style)?;
            } else {
                let sl_idx = line_idx - display_lines.len();
                if sl_idx < streaming_lines.len() {
                    write_line(frame, cols, &streaming_lines[sl_idx], style_streaming())?;
                } else {
                    write_line(frame, cols, "", style_dim())?;
                }
            }
            rendered_history += 1;
        }

        // Fill any remaining history rows if short
        for _ in rendered_history..history_rows {
            write_line(frame, cols, "", style_dim())?;
        }

        // ── Separator ─────────────────────────────────────────────────────────
        let sep = "─".repeat(cols);
        writeln!(frame, "{}{}{}", style_dim(), sep, TerminalStyle::RESET)?;

        // ── Input line ────────────────────────────────────────────────────────
        let prompt_prefix = " ❯ ";
        let max_input_w = cols.saturating_sub(prompt_prefix.len() + 1);
        // Show only the last `max_input_w` chars if the input is very long
        let input_chars: Vec<char> = self.input.chars().collect();
        let visible_start = if input_chars.len() > max_input_w {
            input_chars.len() - max_input_w
        } else {
            0
        };
        let visible: String = input_chars[visible_start..].iter().collect();
        let cursor_visible = self.cursor.saturating_sub(visible_start);

        // Build the input line with a block cursor
        let mut input_line = String::new();
        let visible_chars: Vec<char> = visible.chars().collect();
        for (i, &ch) in visible_chars.iter().enumerate() {
            if i == cursor_visible {
                // Draw cursor block
                let _ = write!(input_line, "{}\x1b[7m{}\x1b[27m", style_input(), ch);
            } else {
                input_line.push(ch);
            }
        }
        // If cursor is at the end, draw a block space
        if cursor_visible >= visible_chars.len() {
            let _ = write!(input_line, "\x1b[7m \x1b[27m");
        }

        if self.is_generating {
            writeln!(
                frame,
                "{} {} {}{}",
                style_dim(),
                "Generating… (Ctrl+C to abort)",
                TerminalStyle::RESET,
                " ".repeat(cols.saturating_sub(32)),
            )?;
        } else {
            writeln!(
                frame,
                "{}{}{}{}",
                style_input(),
                prompt_prefix,
                TerminalStyle::RESET,
                input_line,
            )?;
        }

        // ── Status bar ────────────────────────────────────────────────────────
        let stats_text = if let Some(s) = &self.last_stats {
            format!("{:.1} tok/s  TTFT {:.0}ms", s.generation_tps, s.ttft_ms)
        } else {
            "Ready".to_string()
        };

        let notif_text = if let Some(n) = &self.notification {
            n.clone()
        } else {
            "Ctrl+C: quit  /reset  /help  /stats  PgUp/Dn: scroll".to_string()
        };

        let status = format!(" {} │ {}", stats_text, notif_text);
        let style = if self.notification_error { style_error() } else { style_status() };
        write_line(frame, cols, &truncate_str(&status, cols), style)?;

        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn scroll_to_bottom(&mut self) {
        self.scroll = 0; // 0 means "auto-scroll to bottom" in our renderer
    }

    fn set_notification(&mut self, msg: &str, error: bool) {
        self.notification = Some(msg.to_string());
        self.notification_error = error;
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Write a fixed-width line padded to `cols` with a trailing newline.
fn write_line(frame: &mut TerminalFrame, cols: usize, text: &str, style: TerminalStyle) -> fmt::Result {
    // Measure visible character count (ignore ANSI escape sequences).
    let visible_len = strip_ansi_len(text);
    let padding = cols.saturating_sub(visible_len);
    writeln!(
        frame,
        "{}{}{}{}",
        style,
        text,
        TerminalStyle::RESET,
        " ".repeat(padding)
    )
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
