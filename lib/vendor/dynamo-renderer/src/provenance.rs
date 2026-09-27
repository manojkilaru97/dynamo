// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit output provenance for the supported Super chat template. No request
//! text is replaced or searched to discover its position in the rendered prompt.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use minijinja::{Environment, Error, ErrorKind, Output, State, Value, value::Object};

#[derive(Debug, Default)]
pub struct RenderedPrompt {
    pub text: String,
    pub user_spans: Vec<Range<usize>>,
}

#[derive(Debug, Default)]
struct Trace {
    prompt: RenderedPrompt,
    user: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Capture(Arc<Mutex<Trace>>);
impl Object for Capture {}

impl std::io::Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let text = std::str::from_utf8(bytes).map_err(std::io::Error::other)?;
        let mut trace = self.0.lock().map_err(|_| std::io::Error::other("poisoned provenance capture"))?;
        let start = trace.prompt.text.len();
        trace.prompt.text.push_str(text);
        let end = trace.prompt.text.len();
        if trace.user && start != end {
            if let Some(last) = trace.prompt.user_spans.last_mut()
                && last.end == start
            {
                last.end = end;
            } else {
                trace.prompt.user_spans.push(start..end);
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Capture {
    pub(crate) fn finish(&self) -> anyhow::Result<RenderedPrompt> {
        let mut trace = self.0.lock().map_err(|_| anyhow::anyhow!("poisoned provenance capture"))?;
        Ok(std::mem::take(&mut trace.prompt))
    }
}

#[derive(Debug)]
struct UserContent(Vec<(String, bool)>);
impl Object for UserContent {}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.to_owned())
}

/// Mirrors only render_content's source classification, then checks its bytes
/// against the template result. Generated MM placeholders remain template data.
fn user_content(rendered: String, source: Value) -> Result<Value, Error> {
    let source: serde_json::Value = serde_json::to_value(source)
        .map_err(|_| invalid("cannot inspect Super user content"))?;
    let mut pieces = Vec::new();
    match source {
        serde_json::Value::String(text) => pieces.push((text, true)),
        serde_json::Value::Null => {}
        serde_json::Value::Array(parts) => {
            let mut images = 0;
            let mut videos = 0;
            let mut user_text = String::new();
            for part in &parts {
                match part["type"].as_str().unwrap_or("") {
                    "image" | "image_url" | "input_image" => images += 1,
                    "video" | "video_url" | "input_video" => videos += 1,
                    "text" | "input_text" => {
                        user_text.push_str(part["text"].as_str().unwrap_or(""));
                    }
                    _ => {}
                }
            }
            // These are the supported template's explicit placeholder rules,
            // not a search for the user span in a flattened prompt.
            if user_text.contains("<image>") { images = 0; }
            if user_text.contains("<video>") { videos = 0; }
            let mut image_index = 0;
            for part in &parts {
                match part["type"].as_str().unwrap_or("") {
                    "image" | "image_url" | "input_image" => {
                        image_index += 1;
                        if images > 1 {
                            pieces.push((format!("<image {image_index}><image>"), false));
                        } else if images == 1 {
                            pieces.push(("<image>".into(), false));
                        }
                    }
                    "video" | "video_url" | "input_video" if videos > 0 => {
                        pieces.push(("<video>".into(), false));
                    }
                    "text" | "input_text" => {
                        pieces.push((part["text"].as_str().unwrap_or("").into(), true));
                    }
                    _ => {}
                }
            }
        }
        _ => return Err(invalid("unsupported Super user content type")),
    }
    let joined: String = pieces.iter().map(|(text, _)| text.as_str()).collect();
    if joined != rendered {
        return Err(invalid("Super user-content provenance differs from template output"));
    }
    Ok(Value::from_object(UserContent(pieces)))
}

fn format(out: &mut Output<'_>, state: &State<'_, '_>, value: &Value) -> Result<(), Error> {
    let Some(content) = value.downcast_object_ref::<UserContent>() else {
        return minijinja::escape_formatter(out, state, value);
    };
    let capture_value = state.lookup("__dynamo_output_provenance");
    let capture = capture_value.as_ref().and_then(Value::downcast_object_ref::<Capture>);
    for (text, user) in &content.0 {
        if let Some(capture) = capture {
            capture.0.lock().map_err(|_| invalid("poisoned provenance capture"))?.user = *user;
        }
        out.write_str(text).map_err(|_| invalid("cannot emit Super user content"))?;
        if let Some(capture) = capture {
            capture.0.lock().map_err(|_| invalid("poisoned provenance capture"))?.user = false;
        }
    }
    Ok(())
}

pub(crate) fn configure(env: &mut Environment<'static>) {
    env.add_filter("dynamo_user_data", user_content);
    env.set_formatter(format);
}

/// Adapt a declared template capability, not arbitrary Jinja. The outer emit is
/// tagged after render_content's macro has finished capturing. Changing any of
/// the supported branch syntax requires a new adapter and parity tests.
pub fn enable_super_user_provenance(config: &mut crate::ChatTemplate) -> anyhow::Result<()> {
    let Some(crate::ChatTemplateValue(either::Either::Left(source))) = &mut config.chat_template else {
        anyhow::bail!("Super user provenance requires a single declared chat template");
    };
    const ORIGINAL: &str = r#"        {%- if message.role == "user" and loop.index0 == ns.last_user_idx and medium_effort %}
            {{- content + '\n\n{reasoning effort: efficient}' }}
        {%- else %}
            {{- content }}
        {%- endif %}"#;
    const TRACKED: &str = r#"        {%- if message.role == "user" %}
            {{- content | dynamo_user_data(message.content | default('', true)) }}
            {%- if loop.index0 == ns.last_user_idx and medium_effort %}
                {{- '\n\n{reasoning effort: efficient}' }}
            {%- endif %}
        {%- else %}
            {{- content }}
        {%- endif %}"#;
    anyhow::ensure!(source.matches(ORIGINAL).count() == 1,
        "chat template does not declare the supported Super user-output capability");
    *source = source.replacen(ORIGINAL, TRACKED, 1);
    Ok(())
}
