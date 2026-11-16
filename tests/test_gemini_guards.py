"""Gemini prompt-length and select-retry behavior (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from tenacity import wait_none

from asset_master_match.services import gemini_service
from asset_master_match.services.gemini_service import (
    PromptTooLongError,
    _retry_select,
    generate_embedding,
    generate_json,
)


class _FakeModels:
    def __init__(self, generate_content):
        self._generate_content = generate_content
        self.embed_calls: list[dict] = []

    def generate_content(self, **kwargs):
        return self._generate_content(**kwargs)

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])


class _FakeClient:
    def __init__(self, generate_content):
        self.models = _FakeModels(generate_content)


def _install_client(monkeypatch, generate_content) -> _FakeClient:
    client = _FakeClient(generate_content)
    monkeypatch.setattr(gemini_service, "get_genai_client", lambda: client)
    monkeypatch.setattr(generate_json.retry, "wait", wait_none())
    return client


def test_generate_json_rejects_oversized_prompt(monkeypatch):
    monkeypatch.setattr(gemini_service, "MAX_PROMPT_LENGTH", 50)

    def _should_not_run(**_k):
        raise AssertionError("API must not run when prompt is too long")

    _install_client(monkeypatch, _should_not_run)
    with pytest.raises(PromptTooLongError):
        generate_json("x" * 51, request_id="t")


def test_prompt_too_long_is_not_retried():
    assert _retry_select(PromptTooLongError("too long")) is False


def test_generate_json_retries_503_once(monkeypatch):
    calls = {"n": 0}

    class _Unavailable(Exception):
        code = 503

        def __init__(self) -> None:
            super().__init__("503 UNAVAILABLE")

    def _generate(**_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Unavailable()
        return SimpleNamespace(text='{"candidate_index": 1}')

    _install_client(monkeypatch, _generate)
    assert generate_json("prompt", request_id="t")["candidate_index"] == 1
    assert calls["n"] == 2


def test_generate_json_retries_timeout_once(monkeypatch):
    calls = {"n": 0}

    def _generate(**_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out")
        return SimpleNamespace(text='{"candidate_index": 2}')

    _install_client(monkeypatch, _generate)
    assert generate_json("prompt", request_id="t")["candidate_index"] == 2
    assert calls["n"] == 2


def test_generate_json_does_not_retry_429(monkeypatch):
    calls = {"n": 0}

    class _Quota(Exception):
        code = 429

        def __init__(self) -> None:
            super().__init__("429 RESOURCE_EXHAUSTED")

    def _generate(**_k):
        calls["n"] += 1
        raise _Quota()

    _install_client(monkeypatch, _generate)
    with pytest.raises(_Quota):
        generate_json("prompt", request_id="t")
    assert calls["n"] == 1


def test_generate_embedding_sets_timeout(monkeypatch):
    monkeypatch.setattr(gemini_service, "EMBED_CALL_TIMEOUT_SEC", 15)
    client = _install_client(monkeypatch, lambda **_k: None)
    values = generate_embedding("pump", request_id="t")
    assert values == [0.1, 0.2]
    timeout_ms = client.models.embed_calls[0]["config"].http_options.timeout
    assert timeout_ms == 15000
    assert client.models.embed_calls[0]["config"].http_options.retry_options.attempts == 1
