"""Small shared helpers — no secrets, no I/O."""

from __future__ import annotations

import re
from typing import Any

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Tokens that contain a run of 6+ digits (serials / invoice ids), not short specs like 12V / 65AH.
_LONG_DIGIT_TOKEN = re.compile(r"\S*\d{6,}\S*")


def to_int_or_none(value: Any) -> int | None:
    """Parse an id-like value. Pinecone metadata numbers often arrive as floats (12.0)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def texts_match(a: Any, b: Any) -> bool:
    return normalize_text(a).casefold() == normalize_text(b).casefold()


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def sanitize_prompt_field(value: Any, max_length: int) -> str:
    """Normalize, strip control chars, and truncate for untrusted prompt data."""
    text = normalize_text(value)
    if not text:
        return ""
    text = _CONTROL_CHARS.sub("", text)
    if max_length > 0 and len(text) > max_length:
        return text[:max_length]
    return text


def sanitize_make_model(value: Any, max_length: int) -> str:
    """Sanitize LLM-proposed MakeModel text before returning to callers."""
    return sanitize_prompt_field(value, max_length)


def clean_description_for_embed(
    description: Any,
    *,
    max_length: int = 100,
) -> str:
    """Clean AssetDescription for Pinecone embed only (not for LLM prompts).

    Order: normalize → strip serial/long-digit tokens → collapse whitespace →
    truncate to ``max_length``. Keeps short measurement numbers (12V, 65AH, 30).
    """
    text = normalize_text(description)
    if not text:
        return ""
    text = _LONG_DIGIT_TOKEN.sub(" ", text)
    text = normalize_text(text)
    if max_length > 0 and len(text) > max_length:
        text = text[:max_length].rstrip()
    return text


def build_embed_query(
    asset_name: Any,
    description: Any = None,
    *,
    max_description_length: int = 800,
    max_embed_query_length: int = 1000,
) -> str:
    """Build the text embedded for Pinecone search.

    AssetName is primary and is preserved when applying the combined length cap.
    AssetDescription is optional enrichment (truncated, then appended).
    """
    name = normalize_text(asset_name)
    if not name:
        return ""

    desc = normalize_text(description)
    if not desc:
        if max_embed_query_length > 0 and len(name) > max_embed_query_length:
            return name[:max_embed_query_length]
        return name

    if max_description_length > 0 and len(desc) > max_description_length:
        desc = desc[:max_description_length]

    combined = f"{name}\n{desc}"
    if max_embed_query_length <= 0 or len(combined) <= max_embed_query_length:
        return combined

    # Keep name intact; fill remaining budget with description after a newline.
    if len(name) >= max_embed_query_length:
        return name[:max_embed_query_length]
    budget = max_embed_query_length - len(name) - 1
    if budget <= 0:
        return name
    return f"{name}\n{desc[:budget]}"
