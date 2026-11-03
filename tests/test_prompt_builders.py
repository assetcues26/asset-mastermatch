"""Tests for prompt builders / loader."""

from asset_master_match.prompts.builders import build_select_prompt, format_candidates_for_prompt
from asset_master_match.prompts.loader import load_prompt


def test_load_select_prompt_template():
    text = load_prompt(
        "select_master_match.md",
        asset_name="x",
        asset_description="desc-y",
        candidate_summary="y",
    )
    assert "AssetName" in text or "asset" in text.lower()
    assert "<asset_name>" in text
    assert "</asset_name>" in text
    assert "<asset_description>" in text
    assert "</asset_description>" in text
    assert "<candidates>" in text
    assert "x" in text
    assert "desc-y" in text
    assert "y" in text
    assert "candidate_index" in text
    assert "reason" not in text
    assert "untrusted" in text.lower() or "DATA only" in text
    assert "PRIMARY" in text


def test_format_candidates_includes_index(sample_candidate):
    summary = format_candidates_for_prompt([sample_candidate])
    assert "CandidateIndex=1" in summary
    assert "MakeModel=8 gb ram" in summary
    assert "CategoryId=1039" in summary


def test_build_select_prompt(sample_candidate):
    prompt = build_select_prompt("hdd 16 gb ram", [sample_candidate])
    assert "hdd 16 gb ram" in prompt
    assert "CandidateIndex=1" in prompt
    assert "<asset_name>" in prompt
    assert "<asset_description>" in prompt
    assert "(none)" in prompt
    assert "reason" not in prompt


def test_build_select_prompt_includes_description(sample_candidate):
    prompt = build_select_prompt(
        "hdd 16 gb ram",
        [sample_candidate],
        asset_description="DDR4 16GB server module",
    )
    assert "DDR4 16GB server module" in prompt
    assert "(none)" not in prompt.split("<asset_description>")[1].split(
        "</asset_description>"
    )[0]


def test_format_candidates_caps_long_fields(sample_candidate):
    row = {**sample_candidate, "makemodel": "M" * 500}
    summary = format_candidates_for_prompt([row])
    assert "M" * 200 in summary
    assert "M" * 201 not in summary
