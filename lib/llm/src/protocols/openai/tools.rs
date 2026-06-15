// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, FunctionObject};
use serde_json::{Value, json};
use thiserror::Error;

const DEFAULT_GUIDED_TOOL_ARRAY_MAX_ITEMS: u64 = 16;
const DEFAULT_GUIDED_TOOL_STRING_MAX_LENGTH: u64 = 1024;
const DEFAULT_GUIDED_TOOL_SHORT_TEXT_MAX_LENGTH: u64 = 256;
const DEFAULT_GUIDED_TOOL_LONG_TEXT_MAX_LENGTH: u64 = 8192;
const GUIDED_TOOL_LONG_REQUEST_THRESHOLD: usize = 2048;
const GUIDED_TOOL_LONG_REQUEST_MARGIN: u64 = 512;
const GUIDED_TOOL_LONG_TEXT_FIELDS: &[&str] = &["body", "content", "message"];

/// Errors that can occur when deriving JSON schemas for tool_choice requests.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolChoiceError {
    #[error("tool_choice requires a matching `tools` array")]
    MissingTools,
    #[error("tool `{0}` was not provided in `tools`")]
    ToolNotFound(String),
    #[error("$defs for tool `{0}` must be an object")]
    InvalidDefinitionMap(String),
    #[error("duplicate $defs entry `{0}` has conflicting schemas")]
    ConflictingDefinition(String),
    #[error("tool_choice `required` needs at least one tool definition")]
    EmptyTools,
}

/// Builds the JSON schema enforced by Guided Decoding for the given tool_choice/tools pair.
pub fn get_json_schema_from_tools(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
    request_text_len: Option<usize>,
) -> Result<Option<Value>, ToolChoiceError> {
    let Some(choice) = tool_choice else {
        return Ok(None);
    };

    match choice {
        ChatCompletionToolChoiceOption::None | ChatCompletionToolChoiceOption::Auto => Ok(None),
        ChatCompletionToolChoiceOption::Named(named) => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            let tool = find_tool(tools, &named.function.name)
                .ok_or_else(|| ToolChoiceError::ToolNotFound(named.function.name.clone()))?;
            Ok(Some(bound_guided_tool_schema(
                clone_parameters(&tool.function),
                request_text_len,
            )))
        }
        ChatCompletionToolChoiceOption::Required => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            if tools.is_empty() {
                return Err(ToolChoiceError::EmptyTools);
            }
            build_required_schema(tools, request_text_len).map(Some)
        }
    }
}

/// Builds a vLLM/xgrammar structural-tag constraint matching Qwen-style XML tool calls.
pub fn get_qwen_xml_structural_tag_from_tools(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
    request_text_len: Option<usize>,
) -> Result<Option<Value>, ToolChoiceError> {
    let Some(choice) = tool_choice else {
        return Ok(None);
    };

    match choice {
        ChatCompletionToolChoiceOption::None | ChatCompletionToolChoiceOption::Auto => Ok(None),
        ChatCompletionToolChoiceOption::Named(named) => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            let tool = find_tool(tools, &named.function.name)
                .ok_or_else(|| ToolChoiceError::ToolNotFound(named.function.name.clone()))?;
            Ok(Some(json!({
                "type": "structural_tag",
                "format": {
                    "type": "tags_with_separator",
                    "tags": [qwen_xml_tool_tag(tool, request_text_len)],
                    "triggers": [""],
                    "separator": "",
                    "at_least_one": true,
                    "stop_after_first": true,
                },
            })))
        }
        ChatCompletionToolChoiceOption::Required => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            if tools.is_empty() {
                return Err(ToolChoiceError::EmptyTools);
            }
            if tools.len() == 1 {
                return Ok(Some(json!({
                    "type": "structural_tag",
                    "format": {
                        "type": "tags_with_separator",
                        "tags": [qwen_xml_tool_tag(&tools[0], request_text_len)],
                        "triggers": [""],
                        "separator": "",
                        "at_least_one": true,
                        "stop_after_first": true,
                    },
                })));
            }
            let tags: Vec<Value> = tools
                .iter()
                .map(|tool| qwen_xml_tool_tag(tool, request_text_len))
                .collect();
            if tags.is_empty() {
                return Err(ToolChoiceError::EmptyTools);
            }
            let max_tags = (tags.len() as u64).min(DEFAULT_GUIDED_TOOL_ARRAY_MAX_ITEMS);
            Ok(Some(json!({
                "type": "structural_tag",
                "format": {
                    "type": "repeat",
                    "min": 1,
                    "max": max_tags,
                    "triggers": [""],
                    "content": {
                        "type": "or",
                        "elements": tags,
                    },
                },
            })))
        }
    }
}

/// Returns true for forced named tools whose schema has free-form text payload
/// fields. These are safer with the JSON-schema xgrammar path because qwen-XML
/// structural tags buffer the whole tool call until the text field closes.
pub fn named_tool_choice_has_freeform_text_field(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
) -> bool {
    let Some(ChatCompletionToolChoiceOption::Named(named)) = tool_choice else {
        return false;
    };
    let Some(tools) = tools else {
        return false;
    };
    let Some(tool) = find_tool(tools, &named.function.name) else {
        return false;
    };
    tool.function
        .parameters
        .as_ref()
        .is_some_and(schema_has_freeform_text_field)
}

fn schema_has_freeform_text_field(value: &Value) -> bool {
    let Value::Object(object) = value else {
        return false;
    };

    if let Some(Value::Object(properties)) = object.get("properties") {
        for (name, schema) in properties {
            if GUIDED_TOOL_LONG_TEXT_FIELDS.contains(&name.as_str())
                && value_has_string_type(schema)
                && !value_has_any(schema, &["enum", "const"])
            {
                return true;
            }
            if schema_has_freeform_text_field(schema) {
                return true;
            }
        }
    }

    for key in ["items", "additionalProperties", "anyOf", "oneOf", "allOf"] {
        if let Some(nested) = object.get(key) {
            match nested {
                Value::Array(values) => {
                    if values.iter().any(schema_has_freeform_text_field) {
                        return true;
                    }
                }
                Value::Object(_) => {
                    if schema_has_freeform_text_field(nested) {
                        return true;
                    }
                }
                _ => {}
            }
        }
    }

    false
}

fn value_has_string_type(value: &Value) -> bool {
    let Value::Object(object) = value else {
        return false;
    };
    schema_type_includes(object, "string")
}

fn value_has_any(value: &Value, keys: &[&str]) -> bool {
    let Value::Object(object) = value else {
        return false;
    };
    keys.iter().any(|key| object.contains_key(*key))
}

fn find_tool<'a>(tools: &'a [ChatCompletionTool], name: &str) -> Option<&'a ChatCompletionTool> {
    tools.iter().find(|tool| tool.function.name == name)
}

fn clone_parameters(function: &FunctionObject) -> Value {
    function
        .parameters
        .clone()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
}

fn qwen_xml_tool_tag(tool: &ChatCompletionTool, request_text_len: Option<usize>) -> Value {
    json!({
        "type": "tag",
        "begin": format!("<tool_call>\n<function={}>\n", tool.function.name),
        "content": {
            "type": "json_schema",
            "json_schema": bound_guided_tool_schema(
                clone_parameters(&tool.function),
                request_text_len,
            ),
            "style": "qwen_xml",
        },
        "end": "\n</function>\n</tool_call>",
    })
}

/// Builds a JSON Schema for `tool_choice=required` that enforces an array of tool calls.
///
/// # Schema Structure
///
/// The generated schema looks like:
/// ```json
/// {
///   "type": "array",
///   "minItems": 1,
///   "items": {
///     "type": "object",
///     "anyOf": [
///       {
///         "properties": {
///           "name": {"type": "string", "enum": ["tool1"]},
///           "parameters": { /* tool1's parameter schema */ }
///         },
///         "required": ["name", "parameters"]
///       },
///       {
///         "properties": {
///           "name": {"type": "string", "enum": ["tool2"]},
///           "parameters": { /* tool2's parameter schema */ }
///         },
///         "required": ["name", "parameters"]
///       }
///     ]
///   },
///   "$defs": { /* shared type definitions from all tools */ }
/// }
/// ```
///
/// # $defs Handling
///
/// `$defs` contains shared JSON Schema definitions that can be referenced via `$ref`.
/// For example, if two tools reference a common type:
/// ```json
/// {
///   "$defs": {
///     "Location": {
///       "type": "object",
///       "properties": {
///         "city": {"type": "string"},
///         "country": {"type": "string"}
///       }
///     }
///   }
/// }
/// ```
///
/// We extract `$defs` from each tool's schema and merge them into a global `$defs` map
/// at the root level. If multiple tools define the same type, we verify they match to
/// avoid conflicts.
fn build_required_schema(
    tools: &[ChatCompletionTool],
    request_text_len: Option<usize>,
) -> Result<Value, ToolChoiceError> {
    // Accumulator for all shared type definitions ($defs) across tools
    let mut defs: BTreeMap<String, Value> = BTreeMap::new();
    let mut any_of = Vec::with_capacity(tools.len());

    for tool in tools {
        // Extract parameter schema and its $defs (if any)
        let ParamsAndDefs {
            schema,
            defs: new_defs,
        } = split_defs(&tool.function)?;
        merge_defs(&mut defs, new_defs)?;
        any_of.push(json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name],
                },
                "parameters": schema,
            },
            "required": ["name", "parameters"],
            "additionalProperties": false,
        }));
    }

    // Build the top-level array schema with anyOf constraints
    let mut result = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "anyOf": any_of,
        },
    });

    // Attach the merged $defs at the root level if any were collected
    if !defs.is_empty()
        && let Value::Object(map) = &mut result
    {
        map.insert(
            "$defs".to_string(),
            Value::Object(defs.into_iter().collect()),
        );
    }

    Ok(bound_guided_tool_schema(result, request_text_len))
}

fn bound_guided_tool_schema(schema: Value, request_text_len: Option<usize>) -> Value {
    let mut schema = schema;
    cap_guided_tool_schema(&mut schema, None, request_text_len);
    schema
}

fn schema_type_includes(schema: &serde_json::Map<String, Value>, expected: &str) -> bool {
    match schema.get("type") {
        Some(Value::String(value)) => value == expected,
        Some(Value::Array(values)) => values
            .iter()
            .any(|value| matches!(value, Value::String(s) if s == expected)),
        _ => false,
    }
}

fn cap_guided_tool_schema(
    value: &mut Value,
    field_name: Option<&str>,
    request_text_len: Option<usize>,
) {
    let Value::Object(object) = value else {
        return;
    };

    if schema_type_includes(object, "string")
        && !object.contains_key("maxLength")
        && !object.contains_key("enum")
        && !object.contains_key("const")
    {
        object.insert(
            "maxLength".to_string(),
            json!(guided_tool_string_budget(field_name, request_text_len)),
        );
    }

    if schema_type_includes(object, "array") && !object.contains_key("maxItems") {
        object.insert(
            "maxItems".to_string(),
            json!(DEFAULT_GUIDED_TOOL_ARRAY_MAX_ITEMS),
        );
    }

    if schema_type_includes(object, "object") && !object.contains_key("additionalProperties") {
        object.insert("additionalProperties".to_string(), json!(false));
    }

    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(values)) = object.get_mut(key) {
            for nested in values {
                cap_guided_tool_schema(nested, field_name, request_text_len);
            }
        }
    }

    if let Some(Value::Object(properties)) = object.get_mut("properties") {
        for (name, nested) in properties {
            cap_guided_tool_schema(nested, Some(name.as_str()), request_text_len);
        }
    }

    if let Some(items) = object.get_mut("items") {
        cap_guided_tool_schema(items, field_name, request_text_len);
    }

    if let Some(additional) = object.get_mut("additionalProperties") {
        cap_guided_tool_schema(additional, field_name, request_text_len);
    }

    if let Some(Value::Object(defs)) = object.get_mut("$defs") {
        for nested in defs.values_mut() {
            cap_guided_tool_schema(nested, None, request_text_len);
        }
    }
}

fn guided_tool_string_budget(field_name: Option<&str>, request_text_len: Option<usize>) -> u64 {
    match field_name.unwrap_or_default() {
        "body" | "content" | "message" => guided_tool_text_budget(request_text_len),
        "query" | "sql" => 4096,
        "expression" => 256,
        "subject" | "title" => 512,
        "to" | "from" | "email" => 320,
        _ => DEFAULT_GUIDED_TOOL_STRING_MAX_LENGTH,
    }
}

fn guided_tool_text_budget(request_text_len: Option<usize>) -> u64 {
    let Some(request_text_len) = request_text_len else {
        return DEFAULT_GUIDED_TOOL_SHORT_TEXT_MAX_LENGTH;
    };
    if request_text_len < GUIDED_TOOL_LONG_REQUEST_THRESHOLD {
        return DEFAULT_GUIDED_TOOL_SHORT_TEXT_MAX_LENGTH;
    }
    (request_text_len as u64)
        .saturating_add(GUIDED_TOOL_LONG_REQUEST_MARGIN)
        .clamp(
            DEFAULT_GUIDED_TOOL_SHORT_TEXT_MAX_LENGTH,
            DEFAULT_GUIDED_TOOL_LONG_TEXT_MAX_LENGTH,
        )
}

/// Holds a tool's parameter schema and its extracted $defs (if any).
///
/// When a tool's parameters reference shared types via `$ref`, those types
/// are defined in a `$defs` section within the schema. We extract them separately
/// to merge into a global definitions map.
struct ParamsAndDefs {
    /// The parameter schema with `$defs` removed (if it had one)
    schema: Value,
    /// Extracted `$defs` map, or None if the schema had no definitions
    defs: Option<BTreeMap<String, Value>>,
}

/// Extracts `$defs` from a function's parameter schema, returning both the
/// cleaned schema and the definitions separately.
///
/// # Example
///
/// Input schema:
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "location": {"$ref": "#/$defs/Location"}
///   },
///   "$defs": {
///     "Location": {
///       "type": "object",
///       "properties": {"city": {"type": "string"}}
///     }
///   }
/// }
/// ```
///
/// Returns:
/// - schema: same as input but with `$defs` removed
/// - defs: `Some({"Location": {...}})`
fn split_defs(function: &FunctionObject) -> Result<ParamsAndDefs, ToolChoiceError> {
    let mut schema = clone_parameters(function);
    let defs = match &mut schema {
        Value::Object(obj) => {
            if let Some(value) = obj.remove("$defs") {
                Some(convert_defs(function, value)?)
            } else {
                None
            }
        }
        _ => None,
    };

    Ok(ParamsAndDefs { schema, defs })
}

fn convert_defs(
    function: &FunctionObject,
    defs_value: Value,
) -> Result<BTreeMap<String, Value>, ToolChoiceError> {
    match defs_value {
        Value::Object(map) => Ok(map.into_iter().collect()),
        _ => Err(ToolChoiceError::InvalidDefinitionMap(function.name.clone())),
    }
}

/// Merges definitions from one tool into the global `$defs` accumulator.
///
/// # Conflict Detection
///
/// If two tools define the same type name but with different schemas, we return
/// an error. This ensures consistency across tool definitions.
///
/// # Example
///
/// If `target` contains:
/// ```json
/// {"Location": {"type": "object", "properties": {"city": {"type": "string"}}}}
/// ```
///
/// And we try to merge:
/// ```json
/// {"Location": {"type": "object", "properties": {"city": {"type": "number"}}}}
/// ```
///
/// This will return `ToolChoiceError::ConflictingDefinition("Location")`.
fn merge_defs(
    target: &mut BTreeMap<String, Value>,
    defs: Option<BTreeMap<String, Value>>,
) -> Result<(), ToolChoiceError> {
    let Some(defs) = defs else {
        return Ok(());
    };

    for (name, schema) in defs {
        if let Some(existing) = target.get(&name) {
            if existing != &schema {
                return Err(ToolChoiceError::ConflictingDefinition(name));
            }
        } else {
            target.insert(name, schema);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{ChatCompletionToolChoiceOption, ChatCompletionToolType};

    fn sample_tools() -> Vec<ChatCompletionTool> {
        vec![
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: "add_numbers".to_string(),
                    description: Some("Add two integers".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    })),
                    strict: None,
                },
            },
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location", "unit"],
                    })),
                    strict: None,
                },
            },
        ]
    }

    #[test]
    fn named_choice_returns_parameters() {
        let tools = sample_tools();
        let tool_choice = ChatCompletionToolChoiceOption::Named(
            dynamo_protocols::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_protocols::types::FunctionName {
                    name: "get_weather".to_string(),
                },
            },
        );
        let schema =
            get_json_schema_from_tools(Some(&tool_choice), Some(&tools), None).expect("schema");

        assert_eq!(
            schema.unwrap(),
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "maxLength": 1024},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location", "unit"],
                "additionalProperties": false,
            })
        );
    }

    #[test]
    fn qwen_xml_structural_tag_preserves_tool_parameters() {
        let tools = sample_tools();
        let tool_choice = ChatCompletionToolChoiceOption::Named(
            dynamo_protocols::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_protocols::types::FunctionName {
                    name: "get_weather".to_string(),
                },
            },
        );
        let tag = get_qwen_xml_structural_tag_from_tools(Some(&tool_choice), Some(&tools), None)
            .expect("schema")
            .expect("structural tag");
        let schema = &tag["format"]["tags"][0]["content"]["json_schema"];

        assert_eq!(
            schema["properties"]["location"],
            json!({"type": "string", "maxLength": 1024})
        );
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn qwen_xml_required_structural_tag_allows_bounded_calls() {
        let tools = sample_tools();
        let tag = get_qwen_xml_structural_tag_from_tools(
            Some(&ChatCompletionToolChoiceOption::Required),
            Some(&tools),
            None,
        )
        .expect("schema")
        .expect("structural tag");

        let format = &tag["format"];
        assert_eq!(format["type"], json!("repeat"));
        assert_eq!(format["min"], json!(1));
        assert_eq!(format["max"], json!(tools.len()));
        assert_eq!(format["triggers"], json!([""]));
        assert_eq!(format["content"]["type"], json!("or"));
        assert_eq!(
            format["content"]["elements"].as_array().unwrap().len(),
            tools.len()
        );
        assert_eq!(
            format["content"]["elements"][0]["begin"],
            json!("<tool_call>\n<function=add_numbers>\n")
        );
    }

    #[test]
    fn qwen_xml_required_single_tool_stops_after_first() {
        let tools = vec![sample_tools().remove(0)];
        let tag = get_qwen_xml_structural_tag_from_tools(
            Some(&ChatCompletionToolChoiceOption::Required),
            Some(&tools),
            None,
        )
        .expect("schema")
        .expect("structural tag");

        let format = &tag["format"];
        assert_eq!(format["type"], json!("tags_with_separator"));
        assert_eq!(format["at_least_one"], json!(true));
        assert_eq!(format["stop_after_first"], json!(true));
        assert_eq!(format["separator"], json!(""));
        assert_eq!(format["tags"].as_array().unwrap().len(), 1);
        assert_eq!(
            format["tags"][0]["begin"],
            json!("<tool_call>\n<function=add_numbers>\n")
        );
    }

    #[test]
    fn required_choice_builds_any_of_schema() {
        let tools = sample_tools();
        let schema = get_json_schema_from_tools(
            Some(&ChatCompletionToolChoiceOption::Required),
            Some(&tools),
            None,
        )
        .expect("schema");

        let schema = schema.expect("required schema");
        assert_eq!(schema["type"], "array");
        assert_eq!(schema["minItems"], 1);
        assert_eq!(schema["maxItems"], json!(16));
        assert!(schema["items"]["anyOf"].is_array());

        let any_of = schema["items"]["anyOf"].as_array().unwrap();
        assert_eq!(any_of.len(), 2);
        assert_eq!(schema["items"].get("type"), None);
        assert_eq!(any_of[0]["type"], "object");
        assert_eq!(any_of[0]["additionalProperties"], json!(false));
        assert_eq!(
            any_of[0]["properties"]["name"],
            json!({"type": "string", "enum": ["add_numbers"]})
        );
    }

    #[test]
    fn missing_tool_errors() {
        let tools = sample_tools();
        let tool_choice = ChatCompletionToolChoiceOption::Named(
            dynamo_protocols::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_protocols::types::FunctionName {
                    name: "unknown".to_string(),
                },
            },
        );
        let err = get_json_schema_from_tools(Some(&tool_choice), Some(&tools), None).unwrap_err();
        assert_eq!(err, ToolChoiceError::ToolNotFound("unknown".to_string()));
    }

    #[test]
    fn conflicting_defs_errors() {
        let tool = ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "foo".to_string(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "$defs": {
                        "shared": {"type": "string"}
                    }
                })),
                strict: None,
            },
        };

        let mut tool_with_conflict = tool.clone();
        tool_with_conflict.function.parameters = Some(json!({
            "type": "object",
            "$defs": {
                "shared": {"type": "number"}
            }
        }));

        let tools = vec![tool, tool_with_conflict];
        let err = build_required_schema(&tools, None).unwrap_err();
        assert_eq!(
            err,
            ToolChoiceError::ConflictingDefinition("shared".to_string())
        );
    }

    #[test]
    fn guided_tool_schema_preserves_client_schema() {
        let bounded = bound_guided_tool_schema(
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "body": {"type": "string", "maxLength": 8192},
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                }
            }),
            None,
        );

        assert_eq!(bounded["additionalProperties"], json!(false));
        assert_eq!(
            bounded["properties"]["query"],
            json!({"type": "string", "maxLength": 4096})
        );
        assert_eq!(bounded["properties"]["body"]["maxLength"], json!(8192));
        assert_eq!(
            bounded["properties"]["to"],
            json!({"type": "string", "maxLength": 320})
        );
        assert_eq!(
            bounded["properties"]["subject"],
            json!({"type": "string", "maxLength": 512})
        );
        assert_eq!(bounded["properties"]["tags"]["maxItems"], json!(16));
        assert_eq!(
            bounded["properties"]["tags"]["items"],
            json!({"type": "string", "maxLength": 1024})
        );
    }

    #[test]
    fn guided_tool_schema_bounds_user_fields() {
        let short = bound_guided_tool_schema(
            json!({
                "type": "object",
                "properties": {
                    "body": {"type": "string"},
                    "query": {"type": "string"},
                    "expression": {"type": "string"},
                    "subject": {"type": "string"},
                }
            }),
            Some(64),
        );
        assert_eq!(
            short["properties"]["body"],
            json!({"type": "string", "maxLength": 256})
        );
        assert_eq!(
            short["properties"]["query"],
            json!({"type": "string", "maxLength": 4096})
        );
        assert_eq!(
            short["properties"]["expression"],
            json!({"type": "string", "maxLength": 256})
        );
        assert_eq!(
            short["properties"]["subject"],
            json!({"type": "string", "maxLength": 512})
        );

        let long = bound_guided_tool_schema(
            json!({
                "type": "object",
                "properties": {
                    "body": {"type": "string"},
                }
            }),
            Some(4096),
        );
        assert_eq!(
            long["properties"]["body"],
            json!({"type": "string", "maxLength": 4608})
        );
    }
}
