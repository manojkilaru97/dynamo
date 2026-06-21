// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;
use std::sync::OnceLock;

mod base_parser;
mod gemma4_parser;
mod gpt_oss_parser;
mod granite_parser;
mod minimax_append_think_parser;

// Re-export main types and functions for convenience
pub use base_parser::BasicReasoningParser;
pub use gemma4_parser::Gemma4ReasoningParser;
pub use gpt_oss_parser::GptOssReasoningParser;
pub use granite_parser::GraniteReasoningParser;
pub use minimax_append_think_parser::MiniMaxAppendThinkParser;

/// Repair a stray prefix leaked at the reasoning->answer boundary under
/// speculative decoding (e.g. `{\n{"a":1}`, `{"{"code":1}`, `{ {"a":1}`). When
/// `</think>` plus the first answer token(s) land in one accepted spec batch,
/// a few structural chars can leak ungrammared before the real JSON object.
///
/// Airtight: rewrites only when the original content is NOT valid JSON, the
/// leaked prefix is pure structural junk (`{`, `}`, `"`, `,`, whitespace -- never
/// letters/digits/values, so real content is never truncated) and short, AND the
/// remainder parses as a complete valid JSON value. Returns the repaired string
/// or None. Off the streaming hot path (non-stream aggregation only).
pub fn repair_boundary_brace_leak(content: &str) -> Option<String> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with('{') {
        return None;
    }
    if serde_json::from_str::<serde_json::Value>(trimmed.trim_end()).is_ok() {
        return None; // already valid; do not touch
    }
    const MAX_PREFIX: usize = 8; // a real leak is 1-3 tokens
    // Try each later '{' as the true object start; first one whose tail is valid
    // JSON and whose stripped prefix is pure structural junk wins.
    for (i, c) in trimmed.char_indices() {
        if c != '{' || i == 0 {
            continue;
        }
        if i > MAX_PREFIX {
            break;
        }
        let prefix = &trimmed[..i];
        if !prefix
            .chars()
            .all(|c| matches!(c, '{' | '}' | '"' | ',' | ' ' | '\t' | '\n' | '\r'))
        {
            break;
        }
        let candidate = trimmed[i..].trim_end();
        if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
            return Some(candidate.to_string());
        }
    }
    None
}

#[cfg(test)]
mod brace_leak_tests {
    use super::repair_boundary_brace_leak;

    #[test]
    fn repairs_leaked_prefix_variants() {
        // brace + newline
        assert_eq!(
            repair_boundary_brace_leak("{\n{\"a\":1}").as_deref(),
            Some("{\"a\":1}")
        );
        // brace + newline + space
        assert_eq!(
            repair_boundary_brace_leak("{\n {\"x\":\"y\",\"z\":1}").as_deref(),
            Some("{\"x\":\"y\",\"z\":1}")
        );
        // brace + quote (observed with guided_grammar)
        assert_eq!(
            repair_boundary_brace_leak("{\"{\"code\":\"AB\",\"value\":123}").as_deref(),
            Some("{\"code\":\"AB\",\"value\":123}")
        );
        // brace + space
        assert_eq!(
            repair_boundary_brace_leak("{ {\"a\":1}").as_deref(),
            Some("{\"a\":1}")
        );
        // trailing whitespace tolerated
        assert_eq!(
            repair_boundary_brace_leak("{\n{\"a\":1}\n").as_deref(),
            Some("{\"a\":1}")
        );
    }

    #[test]
    fn leaves_valid_json_untouched() {
        assert_eq!(repair_boundary_brace_leak("{\"a\":1}"), None);
        assert_eq!(repair_boundary_brace_leak("{\"a\":{\"b\":1}}"), None);
        assert_eq!(repair_boundary_brace_leak("{\"a\":1}\n"), None);
    }

    #[test]
    fn leaves_non_repairable_untouched() {
        assert_eq!(repair_boundary_brace_leak("hello world"), None);
        // remainder not valid JSON -> untouched
        assert_eq!(repair_boundary_brace_leak("{ {nested} } trailing"), None);
        assert_eq!(repair_boundary_brace_leak(""), None);
        // prefix contains a real letter (not pure junk) -> untouched
        assert_eq!(repair_boundary_brace_leak("{x{\"a\":1}"), None);
    }
}

/// Kimi-K2/K2.5 tool-call section marker. Shared between the `kimi_k25` reasoning-parser
/// registration and its test fixtures so both stay in sync. Mirrors
/// `KimiK2ParserConfig::default().section_start` in `crate::tool_calling::config`.
pub(crate) const KIMI_K2_TOOL_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";

static REASONING_PARSER_MAP: OnceLock<HashMap<&'static str, ReasoningParserType>> = OnceLock::new();

/// Initialize the global reasoning parser map
fn get_reasoning_parser_map() -> &'static HashMap<&'static str, ReasoningParserType> {
    REASONING_PARSER_MAP.get_or_init(|| {
        let mut map = HashMap::new();
        map.insert("deepseek_r1", ReasoningParserType::DeepseekR1);
        map.insert("basic", ReasoningParserType::Basic);
        map.insert("gpt_oss", ReasoningParserType::GptOss);
        map.insert("qwen3", ReasoningParserType::Qwen);
        // DeepSeek-V4 uses the same `<think>` / `</think>` delimiters as Qwen
        // (confirmed against deepseek-ai/DeepSeek-V4-Pro's encoding_dsv4.py)
        // so it delegates to the same `BasicReasoningParser` config today. We
        // still route through a dedicated `DeepSeekV4` variant rather than
        // hard-aliasing to `Qwen` so future divergence (different special
        // tokens, max-thinking mode, etc.) has a place to land without rippling
        // through Qwen's own config.
        //
        // The three name aliases exist because callers set this via
        // `--dyn-reasoning-parser` / `--reasoning-parser` with whatever string
        // the HF model / vLLM recipe / chat-template author picked. We accept
        // all three separator conventions (snake / kebab / concat) rather than
        // force a single canonical form on users.
        map.insert("deepseek_v4", ReasoningParserType::DeepSeekV4);
        map.insert("deepseek-v4", ReasoningParserType::DeepSeekV4);
        map.insert("deepseekv4", ReasoningParserType::DeepSeekV4);
        map.insert("nemotron_deci", ReasoningParserType::NemotronDeci);
        map.insert("kimi", ReasoningParserType::Kimi);
        map.insert("kimi_k25", ReasoningParserType::KimiK25);
        map.insert("step3", ReasoningParserType::Step3);
        map.insert("mistral", ReasoningParserType::Mistral);
        map.insert("granite", ReasoningParserType::Granite);
        map.insert("nemotron_nano", ReasoningParserType::NemotronV3); // nemotron nano is ...</think>
        map.insert("nemotron3", ReasoningParserType::NemotronV3);
        map.insert("nemotron_v3", ReasoningParserType::NemotronV3);
        map.insert("glm45", ReasoningParserType::NemotronDeci); // GLM-4.5/5 is <think>...</think>, no force_reasoning
        map.insert(
            "minimax_append_think",
            ReasoningParserType::MiniMaxAppendThink,
        );
        // Gemma 4 thinking models: reasoning is wrapped in `<|channel>...<channel|>`
        // with a `thought\n` role label that this parser strips. Pair with
        // `--dyn-tool-call-parser gemma4` for end-to-end Gemma 4 support.
        map.insert("gemma4", ReasoningParserType::Gemma4);
        map.insert("gemma-4", ReasoningParserType::Gemma4);
        map
    })
}

/// Get all available reasoning parser names
pub fn get_available_reasoning_parsers() -> Vec<&'static str> {
    get_reasoning_parser_map().keys().copied().collect()
}

#[derive(Debug, Clone, Default)]
pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

impl ParserResult {
    pub fn get_some_reasoning(&self) -> Option<String> {
        if self.reasoning_text.is_empty() {
            None
        } else {
            Some(self.reasoning_text.clone())
        }
    }

    pub fn get_some_normal_text(&self) -> Option<String> {
        if self.normal_text.is_empty() {
            None
        } else {
            Some(self.normal_text.clone())
        }
    }
}

pub trait ReasoningParser: Send + std::fmt::Debug {
    /// Parses a standalone, non-streaming input chunk. Implementations may reset or ignore
    /// internal streaming state and should return the split of normal vs reasoning text for
    /// this complete input. Marker tokens must not be included in either output.
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult;

    /// Parses a streaming chunk and updates internal state. The return value should be the
    /// delta: only the newly discovered normal and reasoning text attributable to this chunk
    /// (not the cumulative totals). Marker tokens must not be included in either output.
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult;

    /// Override the parser's initial reasoning state. When called with `true`, the parser
    /// starts in reasoning mode without waiting for the start token in the completion stream.
    /// Use this when the chat template already injected the start token (e.g., `<think>`)
    /// into the prompt, so it won't appear in the model's output.
    fn set_in_reasoning(&mut self, _in_reasoning: bool) {
        // Default no-op for parsers that don't support per-request overrides.
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningParserType {
    DeepseekR1,
    Step3,
    Basic,
    GptOss,
    Qwen,
    /// DeepSeek-V4-Pro / V4-Flash. Currently uses the same `<think>` /
    /// `</think>` `BasicReasoningParser` config as Qwen (V4 never appends
    /// `<think>` in the completion — the chat template always pre-injects it,
    /// so the parser starts via `set_in_reasoning(true)` rather than
    /// `force_reasoning`). A dedicated variant keeps future V4-specific
    /// divergence (different delimiters, thinking-effort modes) from leaking
    /// into Qwen's behavior.
    DeepSeekV4,
    NemotronDeci,
    Kimi,
    KimiK25,
    Mistral,
    Granite,
    NemotronV3,
    MiniMaxAppendThink,
    /// Google Gemma 4 thinking models. Custom `<|channel>...<channel|>`
    /// delimiters with a `thought\n` role-label prefix stripped by the parser.
    Gemma4,
}

#[derive(std::fmt::Debug)]
pub struct ReasoningParserWrapper {
    parser: Box<dyn ReasoningParser>,
}

impl ReasoningParser for ReasoningParserWrapper {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.parser.detect_and_parse_reasoning(text, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.parser
            .parse_reasoning_streaming_incremental(text, token_ids)
    }

    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.parser.set_in_reasoning(in_reasoning)
    }
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self) -> ReasoningParserWrapper {
        let basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), false, true);
        let force_reasoning_basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true);
        match self {
            ReasoningParserType::DeepseekR1 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Step3 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Basic => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Qwen => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            // Same `<think>` / `</think>` config as Qwen today; kept as a
            // distinct variant so V4-specific divergence has somewhere to land.
            // See `ReasoningParserType::DeepSeekV4` docstring for rationale.
            ReasoningParserType::DeepSeekV4 => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::NemotronDeci => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Kimi => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "◁think▷".into(),
                    "◁/think▷".into(),
                    false,
                    true,
                )),
            },
            ReasoningParserType::KimiK25 => ReasoningParserWrapper {
                parser: Box::new(
                    BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true)
                        .with_tool_start_token(KIMI_K2_TOOL_SECTION_BEGIN),
                ),
            },
            ReasoningParserType::Mistral => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "[THINK]".into(),
                    "[/THINK]".into(),
                    true,
                    true,
                )),
            },
            ReasoningParserType::GptOss => match GptOssReasoningParser::new() {
                Ok(parser) => ReasoningParserWrapper {
                    parser: Box::new(parser),
                },
                Err(e) => {
                    tracing::warn!(
                        "GptOssReasoningParser could not be initialized, falling back to Basic Reasoning Parser: {e}"
                    );
                    ReasoningParserWrapper {
                        parser: Box::new(BasicReasoningParser::new(
                            "<think>".into(),
                            "</think>".into(),
                            false,
                            true,
                        )),
                    }
                }
            },
            ReasoningParserType::Granite => ReasoningParserWrapper {
                parser: Box::new(GraniteReasoningParser::new()),
            },
            ReasoningParserType::NemotronV3 => ReasoningParserWrapper {
                parser: Box::new(
                    BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true)
                        .with_tool_start_token("<tool_call>"),
                ),
            },
            ReasoningParserType::MiniMaxAppendThink => ReasoningParserWrapper {
                parser: Box::new(MiniMaxAppendThinkParser::new()),
            },
            ReasoningParserType::Gemma4 => ReasoningParserWrapper {
                parser: Box::new(Gemma4ReasoningParser::new()),
            },
        }
    }

    pub fn get_reasoning_parser_from_name(name: &str) -> ReasoningParserWrapper {
        tracing::debug!("Selected reasoning parser: {}", name);

        let parser_map = get_reasoning_parser_map();
        let normalized_name = name.to_lowercase();

        match parser_map.get(normalized_name.as_str()) {
            Some(parser_type) => parser_type.get_reasoning_parser(),
            None => {
                tracing::warn!(
                    parser_name = name,
                    "Unknown reasoning parser type, falling back to Basic Reasoning Parser",
                );
                Self::Basic.get_reasoning_parser()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // registry helper
    fn test_get_available_reasoning_parsers() {
        let parsers = get_available_reasoning_parsers();
        assert!(!parsers.is_empty());
        // Update this list when adding a new parser
        let available_parsers = [
            "deepseek_r1",
            "basic",
            "gpt_oss",
            "qwen3",
            "deepseek_v4",
            "deepseek-v4",
            "deepseekv4",
            "nemotron_deci",
            "kimi",
            "kimi_k25",
            "step3",
            "mistral",
            "granite",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
            "glm45",
            "minimax_append_think",
            "gemma4",
            "gemma-4",
        ];
        for parser in available_parsers {
            assert!(parsers.contains(&parser));
        }
    }

    #[test] // REASONING.batch.1
    fn test_deepseek_v4_detect_and_parse() {
        for parser_name in ["deepseek_v4", "deepseek-v4", "deepseekv4"] {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(parser_name);
            let result = parser.detect_and_parse_reasoning("<think>thinking</think>answer", &[]);
            assert_eq!(result.reasoning_text, "thinking");
            assert_eq!(result.normal_text, "answer");
        }
    }

    #[test] // REASONING.batch.3, REASONING.batch.1
    fn test_deepseek_v4_no_forced_reasoning_without_tags() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        let result = parser.detect_and_parse_reasoning("answer only", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "answer only");
    }

    #[test] // REASONING.stream.3, REASONING.batch.1
    fn test_deepseek_v4_streaming() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");

        let chunks = ["<think>rea", "son</think>answer"];
        let mut reasoning = String::new();
        let mut normal = String::new();

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }

    #[test] // REASONING.batch.1
    fn test_kimi_k25_detect_and_parse() {
        // (description, input, expected_reasoning, expected_normal)
        let cases = [
            (
                "force reasoning: no think tags",
                "no think tags here",
                "no think tags here",
                "",
            ),
            (
                "standard think tags",
                "<think>Let me reason about this.</think>Hello!",
                "Let me reason about this.",
                "Hello!",
            ),
            (
                "empty think block (instant mode)",
                "<think></think>Hello from instant mode!",
                "",
                "Hello from instant mode!",
            ),
            (
                "empty think block with newline",
                "<think>\n</think>Hello from instant mode!",
                "",
                "Hello from instant mode!",
            ),
        ];

        for (desc, input, expected_reasoning, expected_normal) in cases {
            let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
            let result = parser.detect_and_parse_reasoning(input, &[]);
            assert_eq!(
                result.reasoning_text, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(result.normal_text, expected_normal, "FAILED normal: {desc}");
        }
    }

    #[test] // REASONING.stream.3, REASONING.batch.1
    fn test_kimi_k25_streaming_force_reasoning() {
        // Streaming: force_reasoning means tokens before <think> are treated as reasoning
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();

        // First chunk: partial think tag — buffered because it's a prefix of "<think>"
        let r1 = parser.parse_reasoning_streaming_incremental("<thi", &[]);
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        // Second chunk: completes the think tag + reasoning content
        let r2 = parser.parse_reasoning_streaming_incremental("nk>reasoning here", &[]);
        assert_eq!(r2.reasoning_text, "reasoning here");
        assert_eq!(r2.normal_text, "");

        // Third chunk: close tag + normal content
        let r3 = parser.parse_reasoning_streaming_incremental("</think>Hello!", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "Hello!");
    }

    #[test] // REASONING.stream.3, REASONING.batch.1
    fn test_kimi_k25_streaming() {
        // (description, tokens, expected_reasoning, expected_content)
        let cases: Vec<(&str, &[&str], &str, &str)> = vec![
            (
                "complete response",
                &[
                    "<think>",
                    "I need to",
                    " think about",
                    " this carefully.",
                    "</think>",
                    "Bonjour",
                    "!",
                ],
                "I need to think about this carefully.",
                "Bonjour!",
            ),
            (
                "empty think (instant mode)",
                &["<think>", "</think>", "Direct answer."],
                "",
                "Direct answer.",
            ),
        ];

        for (desc, tokens, expected_reasoning, expected_content) in cases {
            let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
            let mut all_reasoning = String::new();
            let mut all_content = String::new();
            for token in tokens {
                let r = parser.parse_reasoning_streaming_incremental(token, &[]);
                all_reasoning.push_str(&r.reasoning_text);
                all_content.push_str(&r.normal_text);
            }
            assert_eq!(
                all_reasoning, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(all_content, expected_content, "FAILED content: {desc}");
        }
    }

    #[test] // registry lookup
    fn test_kimi_k25_parser_lookup_by_name() {
        // Verify the parser can be looked up by name
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k25");
        let result = parser.detect_and_parse_reasoning("<think>thinking</think>answer", &[]);
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(result.normal_text, "answer");
    }

    #[test] // PARSER.fmt.3 — token-spelling differences across model variants
    fn test_kimi_vs_kimi_k25_different_tags() {
        // Kimi (original) uses ◁think▷/◁/think▷, KimiK25 uses <think>/</think>
        let mut kimi = ReasoningParserType::Kimi.get_reasoning_parser();
        let mut kimi_k25 = ReasoningParserType::KimiK25.get_reasoning_parser();

        // Kimi original does NOT parse <think> tags
        let r_kimi = kimi.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(r_kimi.normal_text, "<think>reasoning</think>answer");
        assert_eq!(r_kimi.reasoning_text, "");

        // KimiK25 does parse <think> tags
        let r_k25 = kimi_k25.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(r_k25.reasoning_text, "reasoning");
        assert_eq!(r_k25.normal_text, "answer");
    }

    // Scenario 1: Normal streaming flow with force_reasoning + set_in_reasoning.
    // Simulates the OpenAI path where the preprocessor detects prompt-injected
    // reasoning and calls set_in_reasoning(true). The parser should correctly
    // transition from reasoning to content when </think> arrives.
    #[test] // REASONING.stream.3, REASONING.batch.1
    fn test_nemotron_streaming_with_set_in_reasoning() {
        let mut parser = ReasoningParserType::DeepseekR1.get_reasoning_parser();
        parser.set_in_reasoning(true); // OpenAI path calls this

        let tokens = &["Think", "ing about", " this", ".\n\n", "</think>", "Four"];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "Thinking about this.\n\n");
        assert_eq!(all_content, "Four");
    }

    // Scenario 2: Streaming with force_reasoning but WITHOUT set_in_reasoning.
    // Simulates the Anthropic path bug where thinking_enabled=false and
    // set_in_reasoning is never called. The parser still starts in reasoning
    // mode (force_reasoning=true) but stripped_think_start=false. The </think>
    // boundary must still be detected correctly.
    #[test] // REASONING.stream.3, REASONING.batch.1
    fn test_nemotron_streaming_force_reasoning_without_set_in_reasoning() {
        // DeepseekR1 has force_reasoning=true but we do NOT call set_in_reasoning
        let mut parser = ReasoningParserType::DeepseekR1.get_reasoning_parser();

        let tokens = &["Think", "ing about", " this", ".\n\n", "</think>", "Four"];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "Thinking about this.\n\n");
        assert_eq!(all_content, "Four");
    }

    // Scenario 3: Token-by-token </think> split across chunks.
    // The '<' in '</think>' is a prefix of '<think>'. When stripped_think_start
    // is false, the parser's prefix-check could buffer '<' and interfere with
    // </think> detection. This test verifies the boundary is detected even when
    // </think> arrives as individual characters.
    #[test] // REASONING.stream.3, helper
    fn test_nemotron_streaming_split_end_think_tokens() {
        let mut parser = ReasoningParserType::DeepseekR1.get_reasoning_parser();
        parser.set_in_reasoning(true);

        // Simulate token-by-token arrival including </think> split across chunks
        let tokens = &[
            "reason", "ing", " done", ".", "</", "think", ">", "Hello", " world",
        ];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "reasoning done.");
        assert_eq!(all_content, "Hello world");
    }

    // Scenario: vLLM's Nemotron v3 parser is force-reasoning. The model may
    // begin directly with reasoning text and only emit the closing </think>
    // boundary, or it may emit a full <think>...</think> block. Both forms
    // should split into reasoning_content before </think> and normal content
    // after it.
    #[test] // CASE.10 — vLLM nemotron_v3 parity
    fn test_nemotron_v3_detect_and_parse_vllm_cases() {
        let cases = [
            (
                "without start token",
                "This is a reasoning section</think>This is the rest",
                "This is a reasoning section",
                "This is the rest",
            ),
            (
                "with start token",
                "<think>This is a reasoning section</think>This is the rest",
                "This is a reasoning section",
                "This is the rest",
            ),
        ];

        for (desc, input, expected_reasoning, expected_content) in cases {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name("nemotron_v3");
            let result = parser.detect_and_parse_reasoning(input, &[]);
            assert_eq!(
                result.reasoning_text, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(
                result.normal_text, expected_content,
                "FAILED content: {desc}"
            );
        }
    }

    // Scenario: same vLLM Nemotron v3 contract as the non-streaming test, but
    // exercised as streaming deltas. This verifies the parser keeps state
    // across chunks and handles both prompt-injected reasoning (no opening
    // <think> in output) and explicit <think> output.
    #[test] // CASE.8, CASE.10 — vLLM nemotron_v3 parity
    fn test_nemotron_v3_streaming_vllm_cases() {
        let cases: Vec<(&str, &[&str], &str, &str)> = vec![
            (
                "without start token",
                &[
                    "This is a reasoning section",
                    "</think>",
                    "This is the rest",
                ],
                "This is a reasoning section",
                "This is the rest",
            ),
            (
                "with start token",
                &[
                    "<think>",
                    "This is a reasoning section",
                    "</think>",
                    "This is the rest",
                ],
                "This is a reasoning section",
                "This is the rest",
            ),
        ];

        for (desc, tokens, expected_reasoning, expected_content) in cases {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name("nemotron_v3");
            let mut all_reasoning = String::new();
            let mut all_content = String::new();
            for token in tokens {
                let result = parser.parse_reasoning_streaming_incremental(token, &[]);
                all_reasoning.push_str(&result.reasoning_text);
                all_content.push_str(&result.normal_text);
            }
            assert_eq!(
                all_reasoning, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(all_content, expected_content, "FAILED content: {desc}");
        }
    }

    #[test]
    fn test_nemotron_v3_exits_reasoning_at_tool_call() {
        let input = "Need the tool now<tool_call><function=get_weather></function></tool_call>";
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("nemotron_v3");
        let result = parser.detect_and_parse_reasoning(input, &[]);
        assert_eq!(result.reasoning_text, "Need the tool now");
        assert_eq!(
            result.normal_text,
            "<tool_call><function=get_weather></function></tool_call>"
        );

        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("nemotron_v3");
        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for chunk in [
            "Need the tool now<tool",
            "_call><function=get_weather></function>",
        ] {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            all_reasoning.push_str(&result.reasoning_text);
            all_content.push_str(&result.normal_text);
        }
        assert_eq!(all_reasoning, "Need the tool now");
        assert_eq!(all_content, "<tool_call><function=get_weather></function>");
    }

    // P2-1: V4 production regime where the prompt ends in <think>, so the stream
    // begins INSIDE a reasoning block (no opening <think> sentinel). The caller
    // initializes the parser via set_in_reasoning(true); bytes before </think>
    // must route to reasoning_content, bytes after to normal content.
    #[test]
    fn test_deepseek_v4_streaming_with_set_in_reasoning() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        // Token-by-token stream, starting with raw reasoning (no <think> prefix),
        // </think> in the middle, then normal content.
        let tokens = &[
            "Wei", "gh", "ing ", "options", ".", "</think>", "Bei", "jing", " is", " sunny.",
        ];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "Weighing options.");
        assert_eq!(all_content, "Beijing is sunny.");
    }
}
