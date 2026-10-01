// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicitly opted-in Super user-data encoding. Encoders are immutable and
//! separate, so the ordinary encoding cannot poison the normal tokenizer cache.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use dynamo_renderer::provenance::RenderedPrompt;

use crate::model_card::{ModelDeploymentCard, TokenizerKind};
use crate::tokenizers::traits::Tokenizer;

pub(crate) const MARKERS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

pub(super) fn enabled(mdc: &ModelDeploymentCard) -> bool {
    std::env::var("DYN_SUPER35_USER_DATA_MODEL").is_ok_and(|name| {
        !name.is_empty() && name == mdc.display_name && name.to_ascii_lowercase().contains("super")
    })
}

pub(crate) struct UserDataEncoder {
    ordinary: tokenizers::Tokenizer,
    pub(crate) controls: Arc<std::collections::HashMap<u32, &'static str>>,
}

impl UserDataEncoder {
    pub(super) fn from_mdc(mdc: &ModelDeploymentCard) -> Result<Option<Self>> {
        if !enabled(mdc) { return Ok(None); }
        ensure!(mdc.runtime_config.reasoning_parser.as_deref() == Some("nemotron_v3"),
            "Super user-data policy requires the Nemotron v3 reasoning protocol");
        let Some(TokenizerKind::HfTokenizerJson(file)) = &mdc.tokenizer else {
            anyhow::bail!("Super user-data policy requires an HF tokenizer.json");
        };
        let path = file.path().context("Super user tokenizer must be local")?;
        tracing::info!(model = %mdc.display_name, "enabled explicit Super user-data BPE and token-aware reasoning policy");
        let bytes = std::fs::read(path)?;
        Self::from_json(&bytes).map(Some)
    }

    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self> {
        let mut value: serde_json::Value = serde_json::from_slice(bytes)?;
        let entries = value["added_tokens"].as_array_mut().context("missing added_tokens")?;
        let mut controls = std::collections::HashMap::new();
        for &marker in &MARKERS {
            let entry = entries.iter().find(|v| v["content"].as_str() == Some(marker))
                .with_context(|| format!("missing protocol token {marker}"))?;
            let id = u32::try_from(entry["id"].as_u64().context("invalid token ID")?)?;
            ensure!(entry["lstrip"] == false && entry["rstrip"] == false,
                "control token whitespace stripping is unsupported");
            controls.insert(id, marker);
        }
        entries.retain(|entry| !MARKERS.contains(&entry["content"].as_str().unwrap_or("")));
        // Match ModelDeploymentCard::tokenizer: no implicit truncation/padding.
        value["truncation"] = serde_json::Value::Null;
        value["padding"] = serde_json::Value::Null;
        let ordinary = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&value)?)
            .map_err(anyhow::Error::msg)?;
        Ok(Self { ordinary, controls: Arc::new(controls) })
    }

    pub(crate) fn encode(&self, rendered: &RenderedPrompt, normal: &dyn Tokenizer) -> Result<Vec<u32>> {
        // Preserve exact existing tokenization (including cross-boundary BPE)
        // whenever no user-data span contains a protected protocol spelling.
        let changes_user = rendered.user_spans.iter().any(|span| {
            MARKERS.iter().any(|marker| rendered.text[span.clone()].contains(marker))
        });
        if !changes_user { return Ok(normal.encode(&rendered.text)?.token_ids().to_vec()); }
        let mut ids = Vec::new();
        let mut cursor = 0;
        for span in &rendered.user_spans {
            ensure!(span.start >= cursor && span.end <= rendered.text.len(), "invalid user provenance");
            ids.extend_from_slice(normal.encode(&rendered.text[cursor..span.start])?.token_ids());
            let ordinary = self.ordinary.encode(&rendered.text[span.clone()], false)
                .map_err(anyhow::Error::msg)?;
            ensure!(!ordinary.get_ids().iter().any(|id| self.controls.contains_key(id)),
                "ordinary BPE unexpectedly emitted a protocol control ID");
            ids.extend_from_slice(ordinary.get_ids());
            cursor = span.end;
        }
        ids.extend_from_slice(normal.encode(&rendered.text[cursor..])?.token_ids());
        let decoded: String = normal.decode(&ids, false)?.into();
        ensure!(decoded == rendered.text, "role-aware encoding changed rendered prompt bytes");
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::traits::Encoder;
    use dynamo_renderer::{ChatTemplate, ChatTemplateValue, ContextMixins, PromptFormatter};
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

    /// CPU-only checkpoint tokenizer test; no weights/GPU are loaded. This
    /// exact tokenizer is necessary to prove canonical 315 -> 319 ID parity.
    #[test]
    #[ignore = "requires SUPER35_CPU_MODEL_DIR and SUPER35_CPU_AUDIT_DIR"]
    fn super_user_actual_case03_prompt_parity() -> Result<()> {
        let model = std::path::PathBuf::from(std::env::var("SUPER35_CPU_MODEL_DIR")?);
        let audit = std::path::PathBuf::from(std::env::var("SUPER35_CPU_AUDIT_DIR")?);
        let request: NvCreateChatCompletionRequest = serde_json::from_slice(&std::fs::read(
            audit.join("candidate-p2-h200-case03-r1/03-literal-markers.request.json"))?)?;
        let make = |tracked: bool| -> Result<PromptFormatter> {
            let mut config: ChatTemplate = serde_json::from_slice(&std::fs::read(model.join("tokenizer_config.json"))?)?;
            config.chat_template = Some(ChatTemplateValue(either::Either::Left(
                std::fs::read_to_string(model.join("chat_template.jinja"))?)));
            if tracked { dynamo_renderer::provenance::enable_super_user_provenance(&mut config)?; }
            PromptFormatter::from_parts(config, ContextMixins::default(), true)
        };
        let PromptFormatter::OAI(plain) = make(false)?;
        let PromptFormatter::OAI(tracked) = make(true)?;
        let rendered = tracked.render_with_user_spans(&request)?;
        assert_eq!(plain.render(&request)?, rendered.text);
        let normal = crate::tokenizers::HuggingFaceTokenizer::from_file(model.join("tokenizer.json").to_str().unwrap())?;
        let encoder = UserDataEncoder::from_json(&std::fs::read(model.join("tokenizer.json"))?)?;
        let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(
            audit.join("case03-ordinary-user-span-prompt.json"))?)?;
        let original_ids: Vec<u32> = serde_json::from_value(expected["source_prompt_token_ids"].clone())?;
        let expected_ids: Vec<u32> = serde_json::from_value(expected["prompt_token_ids"].clone())?;
        assert_eq!(normal.encode(&rendered.text)?.token_ids(), original_ids);
        let actual = encoder.encode(&rendered, &normal)?;
        assert_eq!(actual, expected_ids);
        assert_eq!(actual.len(), 319);
        println!("Super CPU unchanged case03: rendered bytes identical; original 315 IDs; canonical role-aware 319 IDs exact");
        Ok(())
    }
}
