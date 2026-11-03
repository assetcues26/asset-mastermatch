"""Gemini prompt-length and rate-limit exception behavior (no network)."""

from __future__ import annotations

import pytest

from asset_master_match.services import gemini_service
from asset_master_match.services.gemini_service import (
    PromptTooLongError,
    RateLimitExceeded,
    generate_json,
)


def test_generate_json_rejects_oversized_prompt(monkeypatch):
    monkeypatch.setattr(gemini_service, "MAX_PROMPT_LENGTH", 50)

    def _should_not_run(*_a, **_k):
        raise AssertionError("rate limit / API must not run when prompt is too long")

    monkeypatch.setattr(gemini_service, "_check_rate_limits", _should_not_run)
    with pytest.raises(PromptTooLongError):
        generate_json("x" * 51, request_id="t")


def test_rate_limit_exceeded_is_not_retried_exception_type():
    # Tenacity only retries TimeoutError / ConnectionError.
    assert not issubclass(RateLimitExceeded, TimeoutError)
    assert not issubclass(RateLimitExceeded, ConnectionError)
    assert not issubclass(PromptTooLongError, TimeoutError)
    assert not issubclass(PromptTooLongError, ConnectionError)
