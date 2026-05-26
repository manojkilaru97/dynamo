# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import re
from typing import Any


DEFAULT_MAX_STRING_LENGTH = int(
    os.environ.get("DYN_SGLANG_GUIDED_JSON_MAX_STRING_LENGTH", "4096")
)
DEFAULT_MAX_ARRAY_ITEMS = int(
    os.environ.get("DYN_SGLANG_GUIDED_JSON_MAX_ARRAY_ITEMS", "16")
)
DEFAULT_MAX_OBJECT_PROPERTIES = int(
    os.environ.get("DYN_SGLANG_GUIDED_JSON_MAX_OBJECT_PROPERTIES", "32")
)
DEFAULT_MAX_REGEX_WILDCARD_LENGTH = int(
    os.environ.get("DYN_SGLANG_GUIDED_REGEX_WILDCARD_LENGTH", "128")
)
DEFAULT_MAX_REGEX_REPEAT_LENGTH = int(
    os.environ.get("DYN_SGLANG_GUIDED_REGEX_REPEAT_LENGTH", "128")
)
DEFAULT_MAX_EBNF_REPEAT_LENGTH = int(
    os.environ.get("DYN_SGLANG_GUIDED_EBNF_REPEAT_LENGTH", "32")
)
DEFAULT_MAX_GUIDED_NEW_TOKENS = int(
    os.environ.get("DYN_SGLANG_GUIDED_MAX_NEW_TOKENS", "4096")
)

_GUIDED_DECODING_KEYS = (
    "json",
    "json_object",
    "choice",
    "regex",
    "grammar",
    "structural_tag",
)

_SGLANG_GUIDED_SAMPLING_KEYS = (
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
)

_JSON_SCHEMA_KEYWORDS = {
    "$defs",
    "$id",
    "$schema",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "default",
    "definitions",
    "description",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "not",
    "oneOf",
    "pattern",
    "patternProperties",
    "prefixItems",
    "properties",
    "required",
    "title",
    "type",
}


def _type_contains(schema: dict[str, Any], type_name: str) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type == type_name
    if isinstance(schema_type, list):
        return type_name in schema_type
    return False


def _is_string_schema(schema: dict[str, Any]) -> bool:
    return _type_contains(schema, "string") or "pattern" in schema


def _is_array_schema(schema: dict[str, Any]) -> bool:
    return _type_contains(schema, "array") or "items" in schema or "prefixItems" in schema


def _is_object_schema(schema: dict[str, Any]) -> bool:
    return (
        _type_contains(schema, "object")
        or "properties" in schema
        or "additionalProperties" in schema
    )


def _bound_regex_wildcards(pattern: str, max_len: int) -> str:
    bounded: list[str] = []
    escaped = False
    in_char_class = False
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if escaped:
            bounded.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            bounded.append(ch)
            escaped = True
            i += 1
            continue
        if ch == "[":
            in_char_class = True
            bounded.append(ch)
            i += 1
            continue
        if ch == "]":
            in_char_class = False
            bounded.append(ch)
            i += 1
            continue
        if not in_char_class and ch == "." and i + 1 < len(pattern):
            quantifier = pattern[i + 1]
            if quantifier == "*":
                bounded.append(f".{{0,{max_len}}}")
                i += 2
                continue
            if quantifier == "+":
                bounded.append(f".{{1,{max_len}}}")
                i += 2
                continue
        if not in_char_class and ch == '"':
            bounded.append('["]')
            i += 1
            continue
        bounded.append(ch)
        i += 1
    return "".join(bounded)


def _bound_regex_open_repeats(pattern: str, max_len: int) -> str:
    """Cap simple unbounded regex repeats that can otherwise run to max_tokens.

    This intentionally handles common single-atom repeats only: character
    classes, escaped atoms, literal atoms, and open ranges. It avoids trying to
    parse full regex syntax in Dynamo.
    """

    atom = r"(\[[^\]\n]+]|\\.|[^\\[\]{}()*+?|])"
    pattern = re.sub(rf"{atom}\+", rf"\1{{1,{max_len}}}", pattern)
    pattern = re.sub(rf"{atom}\*", rf"\1{{0,{max_len}}}", pattern)

    def cap_open_range(match: re.Match[str]) -> str:
        prefix = match.group(1)
        lower = int(match.group(2))
        upper = max(lower, max_len)
        return f"{prefix}{{{lower},{upper}}}"

    return re.sub(rf"{atom}\{{(\d+),\}}", cap_open_range, pattern)


def _bound_ebnf_open_repeats(grammar: str, max_len: int) -> str:
    """Cap simple EBNF char-class repeats without changing prompts/tests.

    xgrammar's EBNF accepts regex-like `[a-z]+`. For structured outputs these
    open repeats can legally continue forever, so rewrite them to one required
    atom plus bounded optional atoms. Keep the rewrite narrow to character
    classes, which is the case used by our public structured-output surface.
    """

    def repeat_plus(match: re.Match[str]) -> str:
        atom = match.group(1)
        return " ".join([atom] + [f"{atom}?" for _ in range(max_len - 1)])

    def repeat_star(match: re.Match[str]) -> str:
        atom = match.group(1)
        return " ".join(f"{atom}?" for _ in range(max_len))

    grammar = re.sub(r"(\[[^\]\n]+])\+", repeat_plus, grammar)
    grammar = re.sub(r"(\[[^\]\n]+])\*", repeat_star, grammar)
    return grammar


def _normalize_json_schema(schema: Any) -> Any:
    if isinstance(schema, bool):
        if not schema:
            return {
                "type": "object",
                "properties": {"__impossible__": {"enum": []}},
                "required": ["__impossible__"],
                "additionalProperties": False,
            }
        return schema
    if not isinstance(schema, dict):
        return schema

    schema = copy.deepcopy(schema)

    if schema == {"not": {"type": "number"}}:
        return {
            "anyOf": [
                {"type": "string"},
                {"type": "object"},
                {"type": "array"},
                {"type": "boolean"},
                {"type": "null"},
            ]
        }

    # Be tolerant of a common "property bag" shape:
    # {"host": {"type": "string"}, "port": {"type": "integer"}, "required": [...]}
    # JSON Schema expects those fields under "properties".
    if "properties" not in schema and isinstance(schema.get("required"), list):
        property_candidates = {
            key: value
            for key, value in schema.items()
            if key not in _JSON_SCHEMA_KEYWORDS and isinstance(value, dict)
        }
        if property_candidates:
            schema["properties"] = property_candidates
            schema.setdefault("type", "object")
            for key in property_candidates:
                schema.pop(key, None)

    for key in ("oneOf", "anyOf", "allOf"):
        values = schema.get(key)
        if isinstance(values, list):
            schema[key] = [_normalize_json_schema(value) for value in values]

    not_schema = schema.get("not")
    if isinstance(not_schema, (dict, bool)):
        schema["not"] = _normalize_json_schema(not_schema)

    for defs_key in ("$defs", "definitions"):
        defs = schema.get(defs_key)
        if isinstance(defs, dict):
            schema[defs_key] = {
                name: _normalize_json_schema(value) for name, value in defs.items()
            }

    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["properties"] = {
            name: _normalize_json_schema(value) for name, value in properties.items()
        }

    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, dict):
        schema["patternProperties"] = {
            name: _normalize_json_schema(value)
            for name, value in pattern_properties.items()
        }

    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, dict):
        schema["additionalProperties"] = _normalize_json_schema(additional_properties)

    items = schema.get("items")
    if isinstance(items, (dict, bool)):
        schema["items"] = _normalize_json_schema(items)
    elif isinstance(items, list):
        schema["items"] = [_normalize_json_schema(value) for value in items]

    prefix_items = schema.get("prefixItems")
    if isinstance(prefix_items, list):
        schema["prefixItems"] = [
            _normalize_json_schema(value) for value in prefix_items
        ]

    if _is_string_schema(schema):
        schema.setdefault("maxLength", DEFAULT_MAX_STRING_LENGTH)
        pattern = schema.get("pattern")
        max_length = schema.get("maxLength")
        if isinstance(pattern, str) and isinstance(max_length, int):
            wildcard_length = min(max_length, DEFAULT_MAX_REGEX_WILDCARD_LENGTH)
            schema["pattern"] = _bound_regex_wildcards(pattern, wildcard_length)

    if _is_array_schema(schema):
        schema.setdefault("maxItems", DEFAULT_MAX_ARRAY_ITEMS)

    if _is_object_schema(schema):
        additional_properties = schema.get("additionalProperties")
        has_pattern_properties = bool(schema.get("patternProperties"))
        if additional_properties is not False or has_pattern_properties:
            schema.setdefault("maxProperties", DEFAULT_MAX_OBJECT_PROPERTIES)

    return schema


def get_guided_decoding_params(
    guided_decoding: object, *, include_structural_tag: bool = False
) -> dict[str, Any]:
    if not isinstance(guided_decoding, dict):
        return {}

    json_schema = guided_decoding.get("json")
    if guided_decoding.get("json_object"):
        json_schema = json_schema or {"type": "object"}
    if json_schema is not None:
        return {"json_schema": json.dumps(_normalize_json_schema(json_schema))}

    if include_structural_tag:
        structural_tag = guided_decoding.get("structural_tag")
        if structural_tag is not None:
            if hasattr(structural_tag, "model_dump"):
                structural_tag = structural_tag.model_dump()
            return {"structural_tag": json.dumps(structural_tag)}

    choice = guided_decoding.get("choice")
    if isinstance(choice, list) and choice:
        escaped = [re.escape(str(value)) for value in choice]
        return {"regex": f"({'|'.join(escaped)})"}

    regex = guided_decoding.get("regex")
    if isinstance(regex, str):
        return {
            "regex": _bound_regex_open_repeats(
                _bound_regex_wildcards(regex, DEFAULT_MAX_REGEX_WILDCARD_LENGTH),
                DEFAULT_MAX_REGEX_REPEAT_LENGTH,
            )
        }

    grammar = guided_decoding.get("grammar")
    if isinstance(grammar, str):
        return {"ebnf": _bound_ebnf_open_repeats(grammar, DEFAULT_MAX_EBNF_REPEAT_LENGTH)}

    return {}


def cap_guided_max_new_tokens(
    params: dict[str, Any], guided_decoding: object
) -> dict[str, Any]:
    has_guided_decoding = isinstance(guided_decoding, dict) and any(
        guided_decoding.get(key) is not None for key in _GUIDED_DECODING_KEYS
    )
    has_guided_sampling_params = any(
        params.get(key) is not None for key in _SGLANG_GUIDED_SAMPLING_KEYS
    )
    if not has_guided_decoding and not has_guided_sampling_params:
        return params

    json_schema = params.get("json_schema")
    if json_schema is not None:
        normalized_json_schema = None
        if isinstance(json_schema, str):
            try:
                normalized_json_schema = json.dumps(
                    _normalize_json_schema(json.loads(json_schema))
                )
            except json.JSONDecodeError:
                normalized_json_schema = None
        elif isinstance(json_schema, (dict, bool)):
            normalized_json_schema = json.dumps(_normalize_json_schema(json_schema))
        if normalized_json_schema is not None and normalized_json_schema != json_schema:
            params = dict(params)
            params["json_schema"] = normalized_json_schema

    max_new_tokens = params.get("max_new_tokens")
    if isinstance(max_new_tokens, int) and max_new_tokens > DEFAULT_MAX_GUIDED_NEW_TOKENS:
        params = dict(params)
        params["max_new_tokens"] = DEFAULT_MAX_GUIDED_NEW_TOKENS
    return params
