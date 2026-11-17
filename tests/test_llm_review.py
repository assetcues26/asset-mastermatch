"""Tests for optional ENABLE_LLM_REVIEW / ENABLE_SUBCATEGORY_GENERATION paths."""

from __future__ import annotations

from asset_master_match.services.match_service import match_one_asset


def test_llm_review_off_ignores_select_payload(
    mock_embedding, mock_pinecone, mock_llm, sample_candidate, monkeypatch
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_LLM_REVIEW",
        False,
    )
    mock_embedding()
    low = {**sample_candidate, "makemodel": "Other", "makemodelid": 1, "score": 0.70}
    high = {
        **sample_candidate,
        "makemodel": "8 gb ram",
        "makemodelid": 10338,
        "score": 0.91,
    }
    mock_pinecone([low, high])
    # Would pick low if LLM ran; with flag OFF top-score wins.
    mock_llm({"candidate_index": 1, "make_model_generated": "N"})
    result = match_one_asset(
        {"AssetName": "hdd 16 gb ram"},
        customer_id=1004,
        request_id="llm-off",
    )
    assert result["MatchFound"] == "Y"
    assert result["MakeModelGenerated"] == "N"
    assert result["SubCategoryGenerated"] == "N"
    assert result["master"]["MakeModelId"] == 10338


def test_llm_review_picks_non_top_candidate(
    mock_embedding, mock_pinecone, monkeypatch, sample_candidate
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_LLM_REVIEW",
        True,
    )
    mock_embedding()
    low = {
        **sample_candidate,
        "makemodel": "Dining Table",
        "makemodelid": 2001,
        "score": 0.70,
    }
    high = {
        **sample_candidate,
        "makemodel": "Server Room Rack",
        "makemodelid": 3416,
        "score": 0.92,
    }
    mock_pinecone([high, low])

    def _llm(prompt, request_id=""):
        assert "Dining Table" in prompt
        assert "Server Room Rack" in prompt
        return {"candidate_index": 2, "make_model_generated": "N"}

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _llm,
    )
    result = match_one_asset(
        {
            "AssetName": "Dining Table 900x900x750",
            "AssetDescription": "Dining Table, White table top",
        },
        customer_id=1004,
        request_id="llm-pick",
    )
    assert result["MatchFound"] == "Y"
    assert result["MakeModelGenerated"] == "N"
    assert result["master"]["MakeModel"] == "Dining Table"
    assert result["master"]["MakeModelId"] == 2001
    assert result["SubCategoryGenerated"] == "N"


def test_llm_review_index_zero_wipes_and_generates_make_model(
    mock_embedding, mock_pinecone, monkeypatch, sample_candidate
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_LLM_REVIEW",
        True,
    )
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_SUBCATEGORY_GENERATION",
        False,
    )
    mock_embedding()
    mock_pinecone([sample_candidate])
    calls: list[str] = []

    def _llm(prompt, request_id=""):
        calls.append(prompt)
        if "candidate_index" in prompt and "Candidates" in prompt:
            return {"candidate_index": 0, "make_model_generated": "N"}
        return {"make_model": "Generic building works"}

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _llm,
    )
    result = match_one_asset(
        {"AssetName": "Building works"},
        customer_id=1158,
        request_id="llm-zero",
    )
    assert result["MatchFound"] == "N"
    assert result["MakeModelGenerated"] == "Y"
    assert result["SubCategoryGenerated"] == "N"
    assert result["master"]["MakeModel"] == "Generic building works"
    assert result["master"]["SubCategory"] is None
    assert result["master"]["CategoryId"] is None
    assert "No acceptable" in result["Message"]
    assert len(calls) == 2


def test_llm_review_fail_falls_back_to_top_score(
    mock_embedding, mock_pinecone, monkeypatch, sample_candidate
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_LLM_REVIEW",
        True,
    )
    mock_embedding()
    mock_pinecone([sample_candidate])

    def _llm(*_a, **_k):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _llm,
    )
    result = match_one_asset(
        {"AssetName": "hdd 16 gb ram"},
        customer_id=1004,
        request_id="llm-fail",
    )
    assert result["MatchFound"] == "Y"
    assert result["MakeModelGenerated"] == "N"
    assert result["master"]["MakeModel"] == "8 gb ram"
    assert result["master"]["MakeModelId"] == 10338
    assert result.get("Error") is not True


def test_llm_review_path_c_generates_when_make_model_na(
    mock_embedding, mock_pinecone, monkeypatch, sample_candidate
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_LLM_REVIEW",
        True,
    )
    mock_embedding()
    row = {**sample_candidate, "makemodel": "N/A", "makemodelid": 1756, "score": 0.85}
    mock_pinecone([row])
    calls: list[dict] = []

    def _llm(prompt, request_id=""):
        if "candidate_index" in prompt and "Candidates" in prompt:
            calls.append({"stage": "select"})
            return {"candidate_index": 1, "make_model_generated": "N"}
        calls.append({"stage": "generate"})
        return {"make_model": "Aluminium step ladder 6 ft"}

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _llm,
    )
    result = match_one_asset(
        {"AssetName": "Aluminium step ladder 6 ft"},
        customer_id=1004,
        request_id="llm-path-c",
    )
    assert result["MatchFound"] == "Y"
    assert result["MakeModelGenerated"] == "Y"
    assert result["master"]["MakeModel"] == "Aluminium step ladder 6 ft"
    assert result["master"]["MakeModelId"] is None
    assert result["master"]["CategoryId"] == 1039
    assert [c["stage"] for c in calls] == ["select", "generate"]


def test_wipe_generates_subcategory_when_enabled(
    mock_embedding, mock_pinecone, monkeypatch
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_LLM_REVIEW",
        False,
    )
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_SUBCATEGORY_GENERATION",
        True,
    )
    mock_embedding()
    mock_pinecone([])
    calls: list[str] = []

    def _llm(prompt, request_id=""):
        calls.append(prompt)
        if "generic SubCategory" in prompt:
            return {"sub_category": "Leasehold Improvement"}
        return {"make_model": "CWIP works"}

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _llm,
    )
    result = match_one_asset(
        {"AssetName": "General Contract Works"},
        customer_id=1158,
        request_id="subcat-on",
    )
    assert result["MatchFound"] == "N"
    assert result["MakeModelGenerated"] == "Y"
    assert result["SubCategoryGenerated"] == "Y"
    assert result["master"]["MakeModel"] == "CWIP works"
    assert result["master"]["SubCategory"] == "Leasehold Improvement"
    assert result["master"]["SubCategoryId"] is None
    assert result["master"]["MakeModelId"] is None
    assert len(calls) == 2


def test_wipe_skips_subcategory_when_disabled(
    mock_embedding, mock_pinecone, mock_llm, monkeypatch
):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.ENABLE_SUBCATEGORY_GENERATION",
        False,
    )
    mock_embedding()
    mock_pinecone([])
    mock_llm({"make_model": "Unknown device"})
    result = match_one_asset(
        {"AssetName": "unknown device"},
        customer_id=1004,
        request_id="subcat-off",
    )
    assert result["MatchFound"] == "N"
    assert result["MakeModelGenerated"] == "Y"
    assert result["SubCategoryGenerated"] == "N"
    assert result["master"]["SubCategory"] is None
