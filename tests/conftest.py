"""Shared fixtures. No live Gemini/Pinecone calls in unit tests."""

from __future__ import annotations

import json
from typing import Any

import pytest


class FakeHttpRequest:
    """Minimal stand-in for azure.functions.HttpRequest."""

    def __init__(self, body: dict[str, Any] | None = None, raw: str | None = None):
        self._body = body
        self._raw = raw

    def get_json(self) -> Any:
        if self._raw is not None:
            return json.loads(self._raw)
        if self._body is None:
            raise ValueError("no body")
        return self._body


@pytest.fixture
def sample_candidate() -> dict[str, Any]:
    return {
        "_id": "1004_null_1107_1039_10067_10338",
        "assetclass": "WATER POLLUTION CONTROL EQUIPMENT",
        "assetclassid": 1107,
        "category": "Machinery",
        "categoryid": 1039,
        "categoryname": "Machinery",
        "customerid": 1004,
        "makemodel": "8 gb ram",
        "makemodelid": 10338,
        "subcategory": "subcategory 1",
        "subcategoryid": 10067,
        "score": 0.91,
    }


@pytest.fixture
def make_http_request():
    def _make(body: dict[str, Any] | None = None, raw: str | None = None) -> FakeHttpRequest:
        return FakeHttpRequest(body=body, raw=raw)

    return _make


@pytest.fixture
def mock_embedding(monkeypatch):
    def _apply(values: list[float] | None = None) -> None:
        vec = values if values is not None else [0.1] * 8

        def _embed(*_a, **_k):
            return vec

        monkeypatch.setattr(
            "asset_master_match.services.match_service.generate_embedding",
            _embed,
        )

    return _apply


@pytest.fixture
def mock_pinecone(monkeypatch):
    def _apply(candidates: list[dict[str, Any]]) -> None:
        monkeypatch.setattr(
            "asset_master_match.services.match_service.query_masterdata",
            lambda *_a, **_k: candidates,
        )

    return _apply


@pytest.fixture
def mock_llm(monkeypatch):
    def _apply(payload: dict[str, Any]) -> None:
        monkeypatch.setattr(
            "asset_master_match.services.match_service.generate_json",
            lambda *_a, **_k: payload,
        )

    return _apply
