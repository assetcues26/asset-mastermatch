"""Gemini JSON parsing helpers (no network)."""

import pytest

from asset_master_match.services.gemini_service import parse_json_response


def test_parse_plain_json():
    data = parse_json_response('{"candidate_index": 1, "make_model_generated": "N"}')
    assert data["candidate_index"] == 1


def test_parse_fenced_json():
    data = parse_json_response('```json\n{"candidate_index": 2, "make_model": "16 gb"}\n```')
    assert data["candidate_index"] == 2
    assert data["make_model"] == "16 gb"


def test_parse_embedded_json():
    data = parse_json_response('Here you go: {"candidate_index": 3, "x": true} thanks')
    assert data["candidate_index"] == 3


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_json_response("   ")
