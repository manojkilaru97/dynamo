// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//based on: https://github.com/EricLBuehler/mistral.rs/blob/d970bb5feb863acf8e8ec90de97e18221fb959f1/mistralrs-core/src/pipeline/chat_template.rs

use std::{collections::HashMap, io};

use chrono::{DateTime, Local};
use either::Either;
use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

pub fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Debug, Deserialize)]
pub struct BeginEndUnkTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

/// Support older tool use patterns where the tool use template was separate from the default/chat template.
/// Modern patterns use a single template with a `tool_use` key, e.g.
///
/// ```jinja
/// {%- if tools is not none and tool_choice is not none %}
/// ```
#[derive(Debug, Deserialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

/// If present, pad_token is usually a single value. Deepseek R1 and it's distill's use a map.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct PadTokenValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
/// Template for chat models including bos/eos/unk as well as the chat template.
pub struct ChatTemplate {
    pub bos_token: Option<BeginEndUnkTok>,
    pub eos_token: Option<BeginEndUnkTok>,
    pub unk_token: Option<BeginEndUnkTok>,

    /// Jinja format [chat templating] for chat completion.
    ///
    /// [chat templating]: https://huggingface.co/docs/transformers/chat_templating
    pub chat_template: Option<ChatTemplateValue>,

    // future
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    legacy: Option<bool>,
    model_max_length: Option<f64>,
    pad_token: Option<PadTokenValue>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: Option<String>,
    truncation_size: Option<String>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn eos_tok(&self) -> Option<String> {
        match self.eos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn bos_tok(&self) -> Option<String> {
        match self.bos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(with = "either::serde_untagged")]
    bos_token_id: Either<u32, Vec<u32>>,
    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<u32, Vec<u32>>,
}

pub fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    if let Ok(indent) = kwargs.get("indent") {
        let mut buf = Vec::new();
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat);
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut serializer).unwrap();
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    } else {
        let mut buf = Vec::new();
        let formatter = PythonDefaultJsonFormatter;
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut serializer).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })?;
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    }
    .map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    .map(|s| {
        // When this filter is used the return value is safe for both HTML and JSON
        let mut rv = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => rv.push_str("\\u003c"),
                '>' => rv.push_str("\\u003e"),
                '&' => rv.push_str("\\u0026"),
                '\'' => rv.push_str("\\u0027"),
                _ => rv.push(c),
            }
        }
        Value::from_safe_string(rv)
    })
}

/// Match Python `json.dumps` defaults used by HuggingFace/Jinja chat templates:
/// comma+space between items and colon+space between object keys and values.
struct PythonDefaultJsonFormatter;

impl serde_json::ser::Formatter for PythonDefaultJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

pub fn strftime_now(format_str: &str) -> Result<Value, Error> {
    let local: DateTime<Local> = Local::now();
    Ok(Value::from_safe_string(
        local.format(format_str).to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::tojson;
    use minijinja::{Environment, context};

    #[test]
    fn tojson_matches_python_default_spacing() {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.add_template(
            "t",
            "{{ array | tojson }}\n{{ object | tojson }}\n{{ nested | tojson }}",
        )
        .unwrap();

        let out = env
            .get_template("t")
            .unwrap()
            .render(context! {
                array => serde_json::json!(["Harry James Potter", "Hermione Jean Granger"]),
                object => serde_json::json!({"a": 1, "b": 2}),
                nested => serde_json::json!({"items": ["x", "y"]}),
            })
            .unwrap();

        assert_eq!(
            out,
            "[\"Harry James Potter\", \"Hermione Jean Granger\"]\n{\"a\": 1, \"b\": 2}\n{\"items\": [\"x\", \"y\"]}"
        );
    }

    #[test]
    fn tojson_preserves_caller_object_order() {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.add_template("t", "{{ tool | tojson }}").unwrap();

        // Tool definitions are prompt text. Keep the incoming JSON order so
        // templates render the same prompt as Hugging Face/vLLM.
        let tool: serde_json::Value = serde_json::from_str(
            r#"{
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Run SQL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "dialect": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }"#,
        )
        .unwrap();

        let out = env
            .get_template("t")
            .unwrap()
            .render(context! { tool => tool })
            .unwrap();

        assert_eq!(
            out,
            r#"{"type": "function", "function": {"name": "execute_sql", "description": "Run SQL", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "dialect": {"type": "string"}}, "required": ["query"]}}}"#
        );
    }
}
