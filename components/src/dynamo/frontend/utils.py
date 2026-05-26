#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared utilities for frontend chat processors (vLLM, SGLang)."""

import re
import uuid
from typing import Any

_MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    """Generate a random 16-character hex UUID."""
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"


def random_call_id() -> str:
    """Generate a random tool call ID in OpenAI format."""
    return f"call_{uuid.uuid4().int & _MASK_64_BITS:016x}"


def worker_warmup() -> bool:
    """Dummy task to ensure a ProcessPoolExecutor worker is fully initialized."""
    return True


class PreprocessError(Exception):
    """Raised by preprocess workers for user-facing errors (e.g., n!=1)."""

    def __init__(self, error_dict: dict[str, Any]):
        self.error_dict = error_dict
        super().__init__(str(error_dict))


# Content part types that carry media URLs, mapped to the key used in the
# multimodal data dict sent to the backend handler.
_MEDIA_CONTENT_TYPES = ("image_url", "audio_url", "video_url")
_HTML_MEDIA_TAG_RE = re.compile(
    r"<\s*(img|image|video|audio)\b[^>]*?\bsrc\s*=\s*([\"'])(.*?)\2[^>]*?/?>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_TO_CONTENT_TYPE = {
    "img": "image_url",
    "image": "image_url",
    "video": "video_url",
    "audio": "audio_url",
}


def _split_html_media_tags(text: str) -> list[dict[str, Any]] | None:
    """Convert supported HTML media tags in text to OpenAI content parts."""
    parts: list[dict[str, Any]] = []
    last = 0
    found = False

    for match in _HTML_MEDIA_TAG_RE.finditer(text):
        if match.start() > last:
            parts.append({"type": "text", "text": text[last : match.start()]})

        url = match.group(3).strip()
        content_type = _HTML_TAG_TO_CONTENT_TYPE[match.group(1).lower()]
        if url:
            parts.append({"type": content_type, content_type: {"url": url}})
            found = True
        else:
            parts.append({"type": "text", "text": match.group(0)})
        last = match.end()

    if not found:
        return None
    if last < len(text):
        parts.append({"type": "text", "text": text[last:]})
    return parts


def normalize_messages_for_multimodal(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize user HTML media tags to standard OpenAI content parts.

    Some clients send Qwen-style ``<video src="..."/>`` or ``<img src="..."/>``
    tags inside text.  The backend only receives real media when the URL is
    represented as an OpenAI content part, so convert just those tags and leave
    unrelated text untouched.
    """
    normalized_messages: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            normalized_messages.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, str):
            parts = _split_html_media_tags(content)
            if parts is None:
                normalized_messages.append(msg)
            else:
                normalized = dict(msg)
                normalized["content"] = parts
                normalized_messages.append(normalized)
            continue

        if not isinstance(content, list):
            normalized_messages.append(msg)
            continue

        changed = False
        normalized_content: list[Any] = []
        for part in content:
            text: str | None = None
            if isinstance(part, dict) and part.get("type") == "text":
                value = part.get("text")
                if isinstance(value, str):
                    text = value
            elif isinstance(part, str):
                text = part

            split_parts = _split_html_media_tags(text) if text is not None else None
            if split_parts is None:
                normalized_content.append(part)
            else:
                normalized_content.extend(split_parts)
                changed = True

        if changed:
            normalized = dict(msg)
            normalized["content"] = normalized_content
            normalized_messages.append(normalized)
        else:
            normalized_messages.append(msg)

    return normalized_messages


def extract_mm_urls(
    messages: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]] | None:
    """Extract multimodal URLs from OpenAI chat completion messages.

    Walks user message content arrays and collects ``image_url``, ``audio_url``,
    and ``video_url`` entries.  Returns them in the format expected by the
    backend handler's ``_extract_multimodal_data()``::

        {
            "image_url": [{"Url": "https://..."}, ...],
            "audio_url": [{"Url": "data:audio/wav;base64,..."}],
        }

    Returns ``None`` if no multimodal content is found.
    """
    mm_data: dict[str, list[dict[str, str]]] = {}

    for msg in normalize_messages_for_multimodal(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in _MEDIA_CONTENT_TYPES:
                continue
            media_value = part.get(part_type)
            if not isinstance(media_value, dict):
                continue
            url = media_value.get("url")
            if isinstance(url, str) and url:
                mm_data.setdefault(part_type, []).append({"Url": url})

    return mm_data or None
