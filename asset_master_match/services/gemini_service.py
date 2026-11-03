"""Gemini text client — timeouts, retries, and simple rate limits."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from asset_master_match.core.config import (
    GEMINI_CALL_TIMEOUT_SEC,
    GEMINI_MAX_CALLS_PER_DAY,
    GEMINI_MAX_CALLS_PER_HOUR,
    GEMINI_MAX_CALLS_PER_MINUTE,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL_NAME,
    MAX_PROMPT_LENGTH,
    clamp_output_tokens,
)

_client: genai.Client | None = None
_client_lock = threading.Lock()

_usage = {
    "daily_calls": 0,
    "hourly_calls": 0,
    "minute_calls": 0,
    "last_reset_day": None,
    "last_reset_hour": None,
    "last_reset_minute": None,
    "lock": threading.Lock(),
}

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class RateLimitExceeded(Exception):
    """Local call-cap exceeded — must not be retried by tenacity."""


class PromptTooLongError(ValueError):
    """Prompt exceeds configured char budget — must not be retried."""


def get_genai_client() -> genai.Client:
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            _client = genai.Client(api_key=api_key)
        return _client


def _check_rate_limits() -> None:
    with _usage["lock"]:
        now = datetime.now()
        if _usage["last_reset_day"] is None:
            _usage["last_reset_day"] = now.day
            _usage["last_reset_hour"] = now.hour
            _usage["last_reset_minute"] = now.minute
        if now.day != _usage["last_reset_day"]:
            _usage["daily_calls"] = 0
            _usage["last_reset_day"] = now.day
        if now.hour != _usage["last_reset_hour"]:
            _usage["hourly_calls"] = 0
            _usage["last_reset_hour"] = now.hour
        if now.minute != _usage["last_reset_minute"]:
            _usage["minute_calls"] = 0
            _usage["last_reset_minute"] = now.minute

        if _usage["daily_calls"] >= GEMINI_MAX_CALLS_PER_DAY:
            raise RateLimitExceeded("GEMINI daily call limit exceeded")
        if _usage["hourly_calls"] >= GEMINI_MAX_CALLS_PER_HOUR:
            raise RateLimitExceeded("GEMINI hourly call limit exceeded")
        if _usage["minute_calls"] >= GEMINI_MAX_CALLS_PER_MINUTE:
            raise RateLimitExceeded("GEMINI per-minute call limit exceeded")

        _usage["daily_calls"] += 1
        _usage["hourly_calls"] += 1
        _usage["minute_calls"] += 1


def parse_json_response(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty Gemini response")
    match = _JSON_BLOCK.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Gemini JSON root must be an object")
    return data


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(GEMINI_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def generate_json(
    prompt: str,
    *,
    model_name: str | None = None,
    max_output_tokens: int | None = None,
    request_id: str = "",
) -> dict[str, Any]:
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise PromptTooLongError(
            f"Prompt length {len(prompt)} exceeds max {MAX_PROMPT_LENGTH}"
        )

    _check_rate_limits()
    model = model_name or GEMINI_MODEL_NAME
    client = get_genai_client()
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=clamp_output_tokens(max_output_tokens),
            http_options=types.HttpOptions(timeout=int(GEMINI_CALL_TIMEOUT_SEC * 1000)),
        ),
    )
    elapsed_ms = (time.time() - start) * 1000
    text = getattr(response, "text", None) or ""
    logging.info(
        "[%s] Gemini select done in %.0fms model=%s chars=%s",
        request_id,
        elapsed_ms,
        model,
        len(text),
    )
    return parse_json_response(text)


def generate_embedding(text: str, request_id: str = "") -> list[float]:
    from asset_master_match.core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

    model = EMBEDDING_MODEL_NAME
    if model.startswith("models/"):
        model = model[len("models/") :]
    client = get_genai_client()
    start = time.time()
    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    elapsed_ms = (time.time() - start) * 1000
    values = list(result.embeddings[0].values)
    logging.info(
        "[%s] Embedding done in %.0fms dims=%s text_len=%s",
        request_id,
        elapsed_ms,
        len(values),
        len(text),
    )
    return values
