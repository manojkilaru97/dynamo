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
/// `separate_media` mirrors the GA template's [`MEDIA_SEPARATOR`] rule.
fn user_content(rendered: String, source: Value, separate_media: Option<bool>) -> Result<Value, Error> {
    let separate_media = separate_media.unwrap_or(false);
    let source: serde_json::Value = serde_json::to_value(source)
        .map_err(|_| invalid("cannot inspect Super user content"))?;
    let mut pieces: Vec<(String, bool)> = Vec::new();
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
            let mut after_text = false;
            let mut emitted = String::new();
            for part in &parts {
                let separate = separate_media && after_text && !emitted.ends_with('\n');
                let placeholder = match part["type"].as_str().unwrap_or("") {
                    "image" | "image_url" | "input_image" => {
                        image_index += 1;
                        match images {
                            0 => None,
                            1 => Some("<image>".to_owned()),
                            _ => Some(format!("<image {image_index}><image>")),
                        }
                    }
                    "video" | "video_url" | "input_video" if videos > 0 => Some("<video>".to_owned()),
                    "text" | "input_text" => {
                        let text = part["text"].as_str().unwrap_or("");
                        after_text |= !text.is_empty();
                        emitted.push_str(text);
                        pieces.push((text.to_owned(), true));
                        None
                    }
                    _ => None,
                };
                if let Some(placeholder) = placeholder {
                    if separate {
                        emitted.push('\n');
                        pieces.push(("\n".to_owned(), false));
                    }
                    emitted.push_str(&placeholder);
                    pieces.push((placeholder, false));
                    after_text = false;
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

const ORIGINAL: &str = r#"        {%- if message.role == "user" and loop.index0 == ns.last_user_idx and medium_effort %}
            {{- content + '\n\n{reasoning effort: efficient}' }}
        {%- else %}
            {{- content }}
        {%- endif %}"#;
const TRACKED: &str = r#"        {%- if message.role == "user" %}
            {{- content | dynamo_user_data(message.content | default('', true)SEPARATE_MEDIA) }}
            {%- if loop.index0 == ns.last_user_idx and medium_effort %}
                {{- '\n\n{reasoning effort: efficient}' }}
            {%- endif %}
        {%- else %}
            {{- content }}
        {%- endif %}"#;
/// The GA template puts one newline between a non-empty text part and the next
/// emitted image or video placeholder unless the content already ends in one.
const MEDIA_SEPARATOR: &str =
    r#"{%- set sep = '\n' if render_ns.after_text and not content_ns.val.endswith('\n') else '' -%}"#;

/// Adapt a declared template capability, not arbitrary Jinja. The outer emit is
/// tagged after render_content's macro has finished capturing. Changing any of
/// the supported branch syntax requires a new adapter and parity tests.
pub fn enable_super_user_provenance(config: &mut crate::ChatTemplate) -> anyhow::Result<()> {
    let Some(crate::ChatTemplateValue(either::Either::Left(source))) = &mut config.chat_template else {
        anyhow::bail!("Super user provenance requires a single declared chat template");
    };
    anyhow::ensure!(source.matches(ORIGINAL).count() == 1,
        "chat template does not declare the supported Super user-output capability");
    let separate_media = match source.matches(MEDIA_SEPARATOR).count() {
        0 => "",
        1 => ", true",
        _ => anyhow::bail!("chat template declares the Super media separator more than once"),
    };
    *source = source.replacen(ORIGINAL, &TRACKED.replacen("SEPARATE_MEDIA", separate_media, 1), 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatTemplate, ContextMixins, PromptFormatter};
    use dynamo_protocols::types::CreateChatCompletionRequest;
    use serde_json::json;

    /// render_content exactly as shipped with the GA (row105) checkpoint.
    const GA_RENDER_CONTENT: &str = r#"{%- macro render_content(content) -%}
    {%- if content is none -%}
        {{- '' -}}
    {%- elif content is string -%}
        {{- content | string -}}
    {%- else -%}
        {%- set text_ns = namespace(val='') -%}
        {%- set counters = namespace(images=0, videos=0) -%}
        {%- for part in content -%}
            {%- if part['type'] == 'image' or part['type'] == 'image_url' or part['type'] == 'input_image' -%}
                {%- set counters.images = counters.images + 1 -%}
            {%- elif part['type'] == 'video' or part['type'] == 'video_url' or part['type'] == 'input_video' -%}
                {%- set counters.videos = counters.videos + 1 -%}
            {%- elif part['type'] == 'text' or part['type'] == 'input_text' -%}
                {%- set text_ns.val = text_ns.val + (part['text'] | default('', true)) -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if '<image>' in text_ns.val -%}
            {%- set counters.images = 0 -%}
        {%- endif -%}
        {%- if '<video>' in text_ns.val -%}
            {%- set counters.videos = 0 -%}
        {%- endif -%}
        {%- set content_ns = namespace(val='') -%}
        {%- set render_ns = namespace(image_index=0, after_text=false) -%}
        {%- for part in content -%}
            {%- set sep = '\n' if render_ns.after_text and not content_ns.val.endswith('\n') else '' -%}
            {%- if part['type'] == 'image' or part['type'] == 'image_url' or part['type'] == 'input_image' -%}
                {%- set render_ns.image_index = render_ns.image_index + 1 -%}
                {%- if counters.images > 0 -%}
                    {%- if counters.images > 1 -%}
                        {%- set content_ns.val = content_ns.val + sep
                            + '<image ' + render_ns.image_index|string + '><image>' -%}
                    {%- else -%}
                        {%- set content_ns.val = content_ns.val + sep + '<image>' -%}
                    {%- endif -%}
                    {%- set render_ns.after_text = false -%}
                {%- endif -%}
            {%- elif part['type'] == 'video' or part['type'] == 'video_url' or part['type'] == 'input_video' -%}
                {%- if counters.videos > 0 -%}
                    {%- set content_ns.val = content_ns.val + sep + '<video>' -%}
                    {%- set render_ns.after_text = false -%}
                {%- endif -%}
            {%- elif part['type'] == 'text' or part['type'] == 'input_text' -%}
                {%- set content_ns.val = content_ns.val + (part['text'] | default('', true)) -%}
                {%- if part['text'] | default('', true) -%}
                    {%- set render_ns.after_text = true -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
        {{- content_ns.val -}}
    {%- endif -%}
{%- endmacro %}"#;
    /// render_content exactly as shipped with the pre-GA (step30) checkpoint.
    const LEGACY_RENDER_CONTENT: &str = r#"{%- macro render_content(content) -%}
    {%- if content is none -%}
        {{- '' -}}
    {%- elif content is string -%}
        {{- content | string -}}
    {%- else -%}
        {%- set text_ns = namespace(val='') -%}
        {%- set counters = namespace(images=0, videos=0) -%}
        {%- for part in content -%}
            {%- if part['type'] == 'image' or part['type'] == 'image_url' or part['type'] == 'input_image' -%}
                {%- set counters.images = counters.images + 1 -%}
            {%- elif part['type'] == 'video' or part['type'] == 'video_url' or part['type'] == 'input_video' -%}
                {%- set counters.videos = counters.videos + 1 -%}
            {%- elif part['type'] == 'text' or part['type'] == 'input_text' -%}
                {%- set text_ns.val = text_ns.val + (part['text'] | default('', true)) -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if '<image>' in text_ns.val -%}
            {%- set counters.images = 0 -%}
        {%- endif -%}
        {%- if '<video>' in text_ns.val -%}
            {%- set counters.videos = 0 -%}
        {%- endif -%}
        {%- set content_ns = namespace(val='') -%}
        {%- set render_ns = namespace(image_index=0) -%}
        {%- for part in content -%}
            {%- if part['type'] == 'image' or part['type'] == 'image_url' or part['type'] == 'input_image' -%}
                {%- set render_ns.image_index = render_ns.image_index + 1 -%}
                {%- if counters.images > 0 -%}
                    {%- if counters.images > 1 -%}
                        {%- set content_ns.val = content_ns.val
                            + '<image ' + render_ns.image_index|string + '><image>' -%}
                    {%- else -%}
                        {%- set content_ns.val = content_ns.val + '<image>' -%}
                    {%- endif -%}
                {%- endif -%}
            {%- elif part['type'] == 'video' or part['type'] == 'video_url' or part['type'] == 'input_video' -%}
                {%- if counters.videos > 0 -%}
                    {%- set content_ns.val = content_ns.val + '<video>' -%}
                {%- endif -%}
            {%- elif part['type'] == 'text' or part['type'] == 'input_text' -%}
                {%- set content_ns.val = content_ns.val + (part['text'] | default('', true)) -%}
            {%- endif -%}
        {%- endfor -%}
        {{- content_ns.val -}}
    {%- endif -%}
{%- endmacro %}"#;
    const MESSAGES: &str = r#"
{%- set medium_effort = medium_effort if medium_effort is defined else False %}
{%- set ns = namespace(last_user_idx = -1) %}
{%- for m in messages %}
  {%- if m["role"] == "user" %}
    {%- set ns.last_user_idx = loop.index0 %}
  {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = render_content(message.content | default('', true)) %}
        {{- '<|im_start|>' + message.role + '\n' }}
"#;

    fn template(render_content: &str) -> ChatTemplate {
        let source = [render_content, MESSAGES, ORIGINAL, "\n        {{- '<|im_end|>\\n' }}\n{%- endfor %}"].concat();
        serde_json::from_value(json!({ "chat_template": source })).unwrap()
    }

    fn formatter(render_content: &str, tracked: bool) -> PromptFormatter {
        let mut config = template(render_content);
        if tracked {
            enable_super_user_provenance(&mut config).unwrap();
        }
        PromptFormatter::from_parts(config, ContextMixins::default(), true).unwrap()
    }

    fn request(content: serde_json::Value) -> CreateChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "super",
            "messages": [{ "role": "user", "content": content }],
        }))
        .unwrap()
    }

    fn text(t: &str) -> serde_json::Value {
        json!({ "type": "text", "text": t })
    }

    fn image() -> serde_json::Value {
        json!({ "type": "image_url", "image_url": { "url": "data:image/png;base64,AA==" } })
    }

    fn video() -> serde_json::Value {
        json!({ "type": "video_url", "video_url": { "url": "data:video/mp4;base64,AA==" } })
    }

    /// Renders with and without provenance; the tracked prompt must be
    /// byte-identical and its user spans must cover exactly `spans`.
    fn assert_tracked(render_content: &str, content: serde_json::Value, body: &str, spans: &[&str]) {
        let req = request(content);
        let PromptFormatter::OAI(plain) = formatter(render_content, false);
        let PromptFormatter::OAI(tracked) = formatter(render_content, true);
        let expected = plain.render(&req).unwrap();
        assert_eq!(expected, format!("<|im_start|>user\n{body}<|im_end|>\n"));
        let rendered = tracked.render_with_user_spans(&req).unwrap();
        assert_eq!(rendered.text, expected);
        let actual: Vec<&str> = rendered.user_spans.iter().map(|s| &rendered.text[s.clone()]).collect();
        assert_eq!(actual, spans);
    }

    #[test]
    fn ga_template_separates_text_from_following_media() {
        let ask = "List the colors in order.";
        assert_tracked(GA_RENDER_CONTENT, json!([text(ask), video()]),
            &format!("{ask}\n<video>"), &[ask]);
        assert_tracked(GA_RENDER_CONTENT, json!([text("How many?"), image(), image(), image()]),
            "How many?\n<image 1><image><image 2><image><image 3><image>", &["How many?"]);
        assert_tracked(GA_RENDER_CONTENT, json!([image(), text("Describe it.")]),
            "<image>Describe it.", &["Describe it."]);
        assert_tracked(GA_RENDER_CONTENT, json!([text("A\n"), image(), text("B"), video(), text("")]),
            "A\n<image>B\n<video>", &["A\n", "B"]);
        assert_tracked(GA_RENDER_CONTENT, json!([text(""), image(), text("<think>x</think>")]),
            "<image><think>x</think>", &["<think>x</think>"]);
    }

    #[test]
    fn ga_template_inline_placeholder_keeps_text_state() {
        // Images are inlined by the user's own <image> tag, so the image part
        // emits nothing and the separator still applies to the later video.
        assert_tracked(GA_RENDER_CONTENT, json!([text("see <image>"), image(), text("then"), video()]),
            "see <image>then\n<video>", &["see <image>then"]);
    }

    #[test]
    fn legacy_template_has_no_media_separator() {
        assert_tracked(LEGACY_RENDER_CONTENT, json!([text("List."), video()]), "List.<video>", &["List."]);
        assert_tracked(LEGACY_RENDER_CONTENT, json!([text("How many?"), image(), image()]),
            "How many?<image 1><image><image 2><image>", &["How many?"]);
        assert_tracked(LEGACY_RENDER_CONTENT, json!([image(), text("Describe it.")]),
            "<image>Describe it.", &["Describe it."]);
    }

    #[test]
    fn adapter_selects_the_declared_separator_rule() {
        let adapted = |render_content: &str| {
            let mut config = template(render_content);
            enable_super_user_provenance(&mut config).unwrap();
            let Some(crate::ChatTemplateValue(either::Either::Left(source))) = config.chat_template else {
                panic!("expected a single chat template");
            };
            source
        };
        assert!(adapted(GA_RENDER_CONTENT).contains("dynamo_user_data(message.content | default('', true), true)"));
        assert!(adapted(LEGACY_RENDER_CONTENT).contains("dynamo_user_data(message.content | default('', true))"));

        let mut twice = template(&[GA_RENDER_CONTENT, MEDIA_SEPARATOR].concat());
        assert!(enable_super_user_provenance(&mut twice).is_err());
    }
}
