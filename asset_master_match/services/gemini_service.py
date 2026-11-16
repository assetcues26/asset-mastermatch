"""Gemini text client — per-call timeouts and one retry for timeouts / 503."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

import httpx
from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from asset_master_match.core.config import (
    EMBED_CALL_TIMEOUT_SEC,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    GEMINI_CALL_TIMEOUT_SEC,
    GEMINI_MODEL_NAME,
    MAX_PROMPT_LENGTH,
    clamp_output_tokens,
)

_client: genai.Client | None = None
_client_lock = threading.Lock()

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# One extra attempt only. 429 stays a single failure so quota errors are not multiplied.
_SELECT_ATTEMPTS = 2
_SELECT_RETRY_BACKOFF_SEC = 1
_RETRYABLE_STATUS_CODES = frozenset({503, 504})


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


def _http_options(timeout_sec: float) -> types.HttpOptions:
    """Per-call timeout. The SDK itself does not retry (attempts=1)."""
    return types.HttpOptions(
        timeout=max(1, int(timeout_sec * 1000)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )


def _retry_select(exc: BaseException) -> bool:
    """Retry a select once on transport timeouts and 503/504. Never on 429."""
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return True
    return getattr(exc, "code", None) in _RETRYABLE_STATUS_CODES


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
    retry=retry_if_exception(_retry_select),
    stop=stop_after_attempt(_SELECT_ATTEMPTS),
    wait=wait_fixed(_SELECT_RETRY_BACKOFF_SEC),
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
        raise PromptTooLongError(f"Prompt length {len(prompt)} exceeds max {MAX_PROMPT_LENGTH}")

    model = model_name or GEMINI_MODEL_NAME
    client = get_genai_client()
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=clamp_output_tokens(max_output_tokens),
            http_options=_http_options(GEMINI_CALL_TIMEOUT_SEC),
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
    model = EMBEDDING_MODEL_NAME
    if model.startswith("models/"):
        model = model[len("models/") :]
    client = get_genai_client()
    start = time.time()
    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
            http_options=_http_options(EMBED_CALL_TIMEOUT_SEC),
        ),
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
