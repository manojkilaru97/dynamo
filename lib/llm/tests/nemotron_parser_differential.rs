// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Same-raw-output differential against the stock vLLM Nemotron parser oracle.
//! The oracle includes every split of structural markers and character streams.
//! Set DYN_NEMOTRON_DIFFERENTIAL_FIXTURES to the generated JSON corpus.
//! Runs with DYN_ENABLE_EXPERIMENTAL_PARSERS_V2 on, as the Super 3 and Super 3.5
//! frontends are deployed: Qwen3-Coder tool calls go through dynamo-parsers-v2.

use std::collections::BTreeMap;
use std::path::PathBuf;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::ParsingOptions;
use dynamo_llm::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};
use serde_json::{Value, json};

fn chunk(text: Option<&str>, finish: Option<&str>) -> NvCreateChatCompletionStreamResponse {
    serde_json::from_value(json!({
        "id": "differential", "object": "chat.completion.chunk", "created": 0,
        "model": "test-model", "choices": [{"index": 0,
            "delta": {"content": text}, "finish_reason": finish}]
    }))
    .unwrap()
}

fn normalized_tools(tools: &Value) -> Value {
    Value::Array(
        tools
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| {
                let arguments = tool["function"]["arguments"].as_str().unwrap_or_default();
                json!({"name": tool["function"]["name"], "arguments":
                    serde_json::from_str::<Value>(arguments)
                        .unwrap_or_else(|_| json!({"invalid_json": arguments}))})
            })
            .collect(),
    )
}

fn regression_corpus() -> Value {
    let mut cases = Vec::new();
    let request = json!({"model": "test-model", "messages": [{"role": "user", "content": "Use exec."}],
        "tool_choice": "auto", "tools": [{"type": "function", "function": {"name": "exec",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]});
    for prefix in ["Reason", "Reason</think>"] {
        for marker in [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<exec>",
            "</exec>",
            "<function=exec>",
            "</function>",
            "</parameter>",
            "<parameter=command>",
        ] {
            let argument = format!("prefix {marker} suffix");
            let expected_argument = match marker {
                "</function>" | "</parameter>" => "prefix ".to_string(),
                "<parameter=command>" => " suffix".to_string(),
                _ => argument.clone(),
            };
            let raw = format!(
                "{prefix}<tool_call><function=exec><parameter=command>\n{argument}\n</parameter></function></tool_call>"
            );
            for streaming in [false, true] {
                for split in 0..=raw.len() {
                    let mut request = request.clone();
                    request["stream"] = json!(streaming);
                    cases.push(json!({"name": format!("literal-{marker}/{prefix}/split-{split}"),
                        "request": request, "chunks": [&raw[..split], &raw[split..]],
                        "expected": {"reasoning": "Reason", "content": "", "finish_reason": "tool_calls",
                            "tools": [{"name": "exec", "arguments": {"command": expected_argument}}]}}));
                }
            }
        }
    }
    let mut typed_request = request.clone();
    typed_request["tools"][0]["function"]["parameters"]["properties"] = json!({
        "command": {"type":"string"}, "number": {"type":"number"},
        "integer": {"type":"integer"}, "boolean": {"type":"boolean"},
        "null": {"type":"null"}, "array": {"type":"array"}, "object": {"type":"object"},
        "nullable": {"type":["string","null"]}
    });
    let scenarios = [
        (
            "types",
            "exec",
            "<parameter=command>42</parameter><parameter=number>3.5</parameter><parameter=integer>42</parameter><parameter=boolean>true</parameter><parameter=null>null</parameter><parameter=array>[1,\"&amp;\",null]</parameter><parameter=object>{\"x\":\"&amp;\"}</parameter><parameter=nullable>null</parameter><parameter=unknown>42</parameter>",
            json!({"command":"42","number":3.5,"integer":42,"boolean":true,"null":null,"array":[1,"&amp;",null],"object":{"x":"&amp;"},"nullable":null,"unknown":"42"}),
        ),
        (
            "unicode_whitespace",
            "exec",
            "<parameter=command>\n 雪😊 &amp; \n\n</parameter>",
            json!({"command":" 雪😊 &amp; \n"}),
        ),
        (
            "empty",
            "exec",
            "<parameter=command></parameter>",
            json!({"command":""}),
        ),
        ("no_parameters", "exec", " ", json!({})),
        (
            "duplicate_last_wins",
            "exec",
            "<parameter=command>first</parameter><parameter=unknown>second</parameter><parameter=command>last</parameter>",
            json!({"command":"last","unknown":"second"}),
        ),
        (
            "unknown_tool",
            "unknown",
            "<parameter=integer>42</parameter><parameter=boolean>true</parameter>",
            json!({"integer":"42","boolean":"true"}),
        ),
        (
            "flexible_parameter_header",
            "exec",
            "< parameter =command>value< / parameter >",
            json!({"command":"value"}),
        ),
    ];
    for (name, tool_name, parameters, arguments) in scenarios {
        let raw = format!(
            "Reason</think><tool_call><function={tool_name}>{parameters}</function></tool_call>"
        );
        for streaming in [false, true] {
            for split in (0..=raw.len()).filter(|index| raw.is_char_boundary(*index)) {
                let mut request = typed_request.clone();
                request["stream"] = json!(streaming);
                cases.push(json!({"name":format!("schema-{name}/split-{split}"), "request":request,
                    "chunks":[&raw[..split],&raw[split..]], "expected":{"reasoning":"Reason","content":"",
                    "finish_reason":"tool_calls","tools":[{"name":tool_name,"arguments":arguments}]}}));
            }
        }
    }
    for raw in [
        "Reason</think><tool_call><function=</function></tool_call>",
        "Reason</think><tool_call><function=exec",
        "Reason</think><tool_call><function=exec<parameter=command>x</parameter></function></tool_call>",
    ] {
        for streaming in [false, true] {
            for split in 0..=raw.len() {
                let mut request = request.clone();
                request["stream"] = json!(streaming);
                cases.push(json!({"name":format!("malformed-no-ghost/{raw}/split-{split}"),"request":request,
                    "chunks":[&raw[..split],&raw[split..]],"expected":{"reasoning":"Reason","content":"","finish_reason":"stop","tools":[]}}));
            }
        }
    }
    let ordinary = cases.clone();
    for mut case in ordinary {
        let mut forced = case.clone();
        forced["name"] = json!(format!(
            "force-nonempty/{}",
            forced["name"].as_str().unwrap()
        ));
        forced["request"]["chat_template_kwargs"] = json!({"force_nonempty_content":true});
        // A malformed invocation still counts as emitted tool-grammar content
        // to the reasoning splitter; its downstream rejection is independent.
        cases.push(forced);
        case["name"] = json!(format!(
            "thinking-disabled/{}",
            case["name"].as_str().unwrap()
        ));
        case["request"]["chat_template_kwargs"] = json!({"enable_thinking":false});
        case["expected"]["content"] = json!(format!(
            "{}{}",
            case["expected"]["reasoning"].as_str().unwrap(),
            case["expected"]["content"].as_str().unwrap()
        ));
        case["expected"]["reasoning"] = json!("");
        cases.push(case);
    }
    for streaming in [false, true] {
        for raw in [
            "Reason",
            "Reason</think>",
            "Reason</think>Answer",
            "Reason</think><exec>data</exec>",
        ] {
            for split in 0..=raw.len() {
                let mut content = raw
                    .split_once("</think>")
                    .map(|(_, content)| content)
                    .filter(|content| !content.is_empty())
                    .unwrap_or("Reason");
                let reasoning = if streaming || content != "Reason" {
                    "Reason"
                } else {
                    ""
                };
                if streaming && raw == "Reason</think>" {
                    content = "";
                }
                let mut request = request.clone();
                request["stream"] = json!(streaming);
                request["chat_template_kwargs"] = json!({"force_nonempty_content":true});
                cases.push(json!({"name":format!("force-fallback/{raw}/split-{split}"),"request":request,
                    "chunks":[&raw[..split],&raw[split..]],"expected":{"reasoning":reasoning,"content":content,"finish_reason":"stop","tools":[]}}));
            }
        }
    }
    json!({"reference": {"vllm_version": "0.29.0", "parser": "stock structural grammar plus lossless nonstructural-argument invariants", "intentional_improvements": ["preserve argument </think>", "emit valid final JSON for duplicate/structurally terminated parameters", "thinking=false consistently absorbs structural </think> outside tools in both modes"]}, "cases": cases})
}

#[tokio::test]
async fn stock_vllm_nemotron_raw_output_differential() {
    // SAFETY: this binary has a single test and sets the switch before any
    // parser reads it (the flag is read once per process).
    unsafe { std::env::set_var("DYN_ENABLE_EXPERIMENTAL_PARSERS_V2", "1") };
    let corpus: Value = if let Ok(path) = std::env::var("DYN_NEMOTRON_DIFFERENTIAL_FIXTURES") {
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    } else {
        regression_corpus()
    };
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    mdc.runtime_config.reasoning_parser = Some("nemotron_v3".to_string());
    mdc.runtime_config.tool_call_parser = Some("qwen3_coder".to_string());
    let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
    let mut failures = Vec::new();
    let mut passed = 0;
    let mut reference_errors = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if !case["expected"]["error"].is_null() {
            reference_errors += 1;
            continue;
        }
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(case["request"].clone()).unwrap();
        let chunks: Vec<_> = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| chunk(text.as_str(), None))
            .chain(std::iter::once(chunk(None, Some("stop"))))
            .map(Annotated::from_data)
            .collect();
        let prompt_injected = case["request"]["chat_template_kwargs"]["enable_thinking"] != false;
        let output = preprocessor
            .postprocessor_parsing_stream(stream::iter(chunks), &request, prompt_injected, false)
            .unwrap();
        let actual = if case["request"]["stream"] == true {
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut tools: BTreeMap<u64, Value> = BTreeMap::new();
            let mut finishes = Vec::new();
            for annotated in Box::pin(output).collect::<Vec<_>>().await {
                let Some(data) = annotated.data else { continue };
                let data = serde_json::to_value(data).unwrap();
                for choice in data["choices"].as_array().unwrap() {
                    let delta = &choice["delta"];
                    reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
                    content.push_str(delta["content"].as_str().unwrap_or_default());
                    for tool in delta["tool_calls"].as_array().into_iter().flatten() {
                        let entry = tools
                            .entry(tool["index"].as_u64().unwrap())
                            .or_insert_with(|| json!({"function": {"name": "", "arguments": ""}}));
                        for field in ["name", "arguments"] {
                            if let Some(fragment) = tool["function"][field].as_str() {
                                let combined = format!(
                                    "{}{fragment}",
                                    entry["function"][field].as_str().unwrap()
                                );
                                entry["function"][field] = json!(combined);
                            }
                        }
                    }
                    if !choice["finish_reason"].is_null() {
                        finishes.push(choice["finish_reason"].clone());
                    }
                }
            }
            json!({"reasoning": reasoning, "content": content,
                "tools": normalized_tools(&json!(tools.into_values().collect::<Vec<_>>())),
                "finish_reason": if finishes.len() == 1 { finishes[0].clone() } else { json!(finishes) }})
        } else {
            let options = ParsingOptions::new(
                Some("qwen3_coder".to_string()),
                Some("nemotron_v3".to_string()),
            )
            .with_experimental_v2_batch_eligible(true);
            match NvCreateChatCompletionResponse::from_annotated_stream(output, options).await {
                Ok(response) => {
                    let response = serde_json::to_value(response).unwrap();
                    let choice = &response["choices"][0];
                    json!({"reasoning": choice["message"]["reasoning_content"].as_str().unwrap_or_default(),
                        "content": choice["message"]["content"].as_str().unwrap_or_default(),
                        "tools": normalized_tools(&choice["message"]["tool_calls"]),
                        "finish_reason": choice["finish_reason"]})
                }
                Err(error) => json!({"error": error.to_string()}),
            }
        };
        if actual == case["expected"] {
            passed += 1;
        } else {
            failures.push(
                json!({"name": case["name"], "stream": case["request"]["stream"],
                "chunks": case["chunks"], "expected": case["expected"], "actual": actual,
                "fields": (["reasoning", "content", "tools", "finish_reason"].into_iter()
                    .filter(|key| actual[*key] != case["expected"][*key]).collect::<Vec<_>>())}),
            );
        }
    }
    let report = json!({"reference": corpus["reference"], "passed": passed,
        "failed": failures.len(), "reference_errors": reference_errors, "failures": failures});
    let report_path = std::env::var("DYN_NEMOTRON_DIFFERENTIAL_REPORT").unwrap_or_default();
    if !report_path.is_empty() {
        std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    eprintln!(
        "Differential: passed={passed} failed={} reference_errors={reference_errors}; report={report_path}",
        failures.len()
    );
    if std::env::var("DYN_NEMOTRON_DIFF_REPORT_ONLY").as_deref() != Ok("1") {
        assert!(
            failures.is_empty(),
            "stock-vLLM semantic differences; see {report_path}"
        );
    }
}
