"""HTTP + orchestration tests with Gemini/Pinecone mocked."""

from __future__ import annotations

import json
import time

from asset_master_match.core.config import MAX_ASSET_NAME_LENGTH
from asset_master_match.services import match_service
from asset_master_match.services.match_service import (
    MSG_TIMED_OUT,
    match_one_asset,
    run_asset_master_match,
)


def _body(resp) -> dict:
    return json.loads(resp.get_body().decode("utf-8"))


def test_http_rejects_missing_customer(make_http_request):
    resp = run_asset_master_match(make_http_request({"assets": [{"AssetName": "x"}]}))
    assert resp.status_code == 400
    assert "CustomerId" in _body(resp)["error"]


def test_http_rejects_empty_assets(make_http_request):
    resp = run_asset_master_match(make_http_request({"CustomerId": 1004, "assets": []}))
    assert resp.status_code == 400


def test_http_rejects_too_many_assets(make_http_request, monkeypatch):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.MAX_ASSETS_PER_REQUEST",
        2,
    )
    resp = run_asset_master_match(
        make_http_request(
            {
                "CustomerId": 1004,
                "assets": [
                    {"AssetName": "a"},
                    {"AssetName": "b"},
                    {"AssetName": "c"},
                ],
            }
        )
    )
    assert resp.status_code == 400
    assert "At most" in _body(resp)["error"]


def test_http_rejects_non_dict_asset(make_http_request):
    resp = run_asset_master_match(
        make_http_request(
            {
                "CustomerId": 1004,
                "assets": [
                    {"AssetName": "ok"},
                    "not-an-object",
                ],
            }
        )
    )
    assert resp.status_code == 400
    assert "assets[1] must be an object" in _body(resp)["error"]


def test_http_rejects_requests_missing_customer(make_http_request):
    resp = run_asset_master_match(
        make_http_request({"requests": [{"assets": [{"AssetName": "x"}]}]})
    )
    assert resp.status_code == 400
    assert "CustomerId" in _body(resp)["error"]


def test_http_rejects_too_many_assets_across_customers(make_http_request, monkeypatch):
    monkeypatch.setattr(
        "asset_master_match.services.match_service.MAX_ASSETS_PER_REQUEST",
        3,
    )
    resp = run_asset_master_match(
        make_http_request(
            {
                "requests": [
                    {
                        "CustomerId": 1001,
                        "assets": [{"AssetName": "a"}, {"AssetName": "b"}],
                    },
                    {
                        "CustomerId": 1004,
                        "assets": [{"AssetName": "c"}, {"AssetName": "d"}],
                    },
                ]
            }
        )
    )
    assert resp.status_code == 400
    assert "At most" in _body(resp)["error"]


def test_http_rejects_invalid_json(make_http_request):
    resp = run_asset_master_match(make_http_request(raw="{not-json"))
    assert resp.status_code == 400


def test_match_one_missing_asset_name():
    result = match_one_asset({}, customer_id=1004, request_id="t1")
    assert result["MatchFound"] == "N"
    assert result.get("Error") is True
    assert "AssetName" in result["Message"]


def test_match_one_asset_name_too_long():
    result = match_one_asset(
        {"AssetName": "x" * (MAX_ASSET_NAME_LENGTH + 1)},
        customer_id=1004,
        request_id="t1b",
    )
    assert result["MatchFound"] == "N"
    assert result.get("Error") is True
    assert "maximum length" in result["Message"]


def test_match_one_no_pinecone_hits(mock_embedding, mock_pinecone, mock_llm):
    mock_embedding()
    mock_pinecone([])
    mock_llm({"make_model": "Unknown device"})
    result = match_one_asset(
        {"AssetName": "unknown device"},
        customer_id=1004,
        request_id="t2",
    )
    assert result["MatchFound"] == "N"
    assert result["MakeModelGenerated"] == "Y"
    assert result["master"]["MakeModel"] == "Unknown device"
    assert result["master"]["MakeModelId"] is None
    assert result["master"]["AssetClassId"] is None
    assert "No master data" in result["Message"]


def test_match_one_success_no_llm_reason(mock_embedding, mock_pinecone, mock_llm, sample_candidate):
    mock_embedding()
    mock_pinecone([sample_candidate])
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "16 gb ram",
            "make_model_generated": "Y",
            "reason": "capacity from name",  # ignored if model still sends it
        }
    )
    result = match_one_asset(
        {
            "AssetName": "hdd 16 gb ram",
            "CategoryId": 1039,
            "AssetClassId": 1107,
            "Cost": 50000,
        },
        customer_id=1004,
        request_id="t3",
    )
    assert result["MatchFound"] == "Y"
    assert result["MakeModelGenerated"] == "N"
    assert result["CustomerId"] == 1004
    assert result["Cost"] == 50000
    assert result["master"]["MakeModel"] == "8 gb ram"
    assert result["master"]["MakeModelId"] == 10338
    assert result["master"]["CategoryId"] == 1039
    assert result["Message"] is None


def test_match_one_embeds_name_plus_description(
    mock_pinecone, mock_llm, sample_candidate, monkeypatch
):
    seen: list[str] = []

    def _embed(text, request_id=""):
        seen.append(text)
        return [0.1] * 8

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_embedding",
        _embed,
    )
    mock_pinecone([sample_candidate])
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "N",
        }
    )
    result = match_one_asset(
        {
            "AssetName": "hdd 16 gb ram",
            "AssetDescription": "DDR4 module for rack server",
            "Cost": 100,
        },
        customer_id=1004,
        request_id="t3desc",
    )
    assert result["MatchFound"] == "Y"
    assert result["AssetDescription"] == "DDR4 module for rack server"
    # A strong name hit does not also embed the description.
    assert seen == ["hdd 16 gb ram"]


def test_match_one_embeds_description_when_name_hit_is_weak(
    mock_pinecone, mock_llm, sample_candidate, monkeypatch
):
    seen: list[str] = []

    def _embed(text, request_id=""):
        seen.append(text)
        return [0.1] * 8

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_embedding",
        _embed,
    )
    mock_pinecone([{**sample_candidate, "score": 0.66}])
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "N",
        }
    )
    result = match_one_asset(
        {
            "AssetName": "IIAT - 1200x650x700-1100ht",
            "AssetDescription": "LINEAR WORKSTATION",
        },
        customer_id=1158,
        request_id="t3weakdesc",
    )
    assert result["MatchFound"] == "Y"
    assert seen == [
        "IIAT - 1200x650x700-1100ht",
        "IIAT - 1200x650x700-1100ht\nLINEAR WORKSTATION",
    ]


def test_match_one_without_description_embeds_name_only(
    mock_pinecone, mock_llm, sample_candidate, monkeypatch
):
    seen: list[str] = []

    def _embed(text, request_id=""):
        seen.append(text)
        return [0.1] * 8

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_embedding",
        _embed,
    )
    mock_pinecone([sample_candidate])
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "N",
        }
    )
    result = match_one_asset(
        {"AssetName": "hdd 16 gb ram"},
        customer_id=1004,
        request_id="t3nodesc",
    )
    assert result["MatchFound"] == "Y"
    assert "AssetDescription" not in result
    assert seen == ["hdd 16 gb ram"]


def test_match_one_select_error_hides_exception(mock_embedding, mock_pinecone, monkeypatch):
    mock_embedding()
    mock_pinecone([{"customerid": 1004, "makemodel": "N/A", "score": 0.8}])

    def _boom(*_a, **_k):
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _boom,
    )
    result = match_one_asset(
        {"AssetName": "pump"},
        customer_id=1004,
        request_id="t4",
    )
    assert result["MatchFound"] == "N"
    assert result.get("Error") is True
    assert result["Message"] == "Failed to select master match"
    assert "GEMINI_API_KEY" not in result["Message"]
    assert "ErrorCode" not in result


def test_match_one_embed_unavailable(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_embedding",
        _boom,
    )
    result = match_one_asset(
        {"AssetName": "pump"},
        customer_id=1004,
        request_id="t4b",
    )
    assert result["MatchFound"] == "N"
    assert result.get("Error") is True
    assert result["Message"] == "Embedding service unavailable"
    assert "503" not in result["Message"]


def test_match_one_select_unavailable(mock_embedding, mock_pinecone, monkeypatch):
    mock_embedding()
    mock_pinecone([{"customerid": 1004, "makemodel": "N/A", "score": 0.8}])

    def _boom(*_a, **_k):
        raise RuntimeError("503 UNAVAILABLE high demand")

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _boom,
    )
    result = match_one_asset(
        {"AssetName": "pump"},
        customer_id=1004,
        request_id="t4c",
    )
    assert result["Message"] == "Gemini selection unavailable"


def test_match_one_select_rate_limited(mock_embedding, mock_pinecone, monkeypatch):
    mock_embedding()
    mock_pinecone([{"customerid": 1004, "makemodel": "NA", "score": 0.8}])

    def _boom(*_a, **_k):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    monkeypatch.setattr(
        "asset_master_match.services.match_service.generate_json",
        _boom,
    )
    result = match_one_asset(
        {"AssetName": "pump"},
        customer_id=1004,
        request_id="t4d",
    )
    assert result["Message"] == "Gemini rate limit hit"


def test_match_one_pinecone_unavailable(mock_embedding, monkeypatch):
    mock_embedding()

    def _boom(*_a, **_k):
        raise ValueError("PINECONE_API_KEY environment variable not set")

    monkeypatch.setattr(
        "asset_master_match.services.match_service.query_masterdata",
        _boom,
    )
    result = match_one_asset(
        {"AssetName": "pump"},
        customer_id=1004,
        request_id="t4e",
    )
    assert result["Message"] == "Master data search unavailable"
    assert "PINECONE_API_KEY" not in result["Message"]


def test_match_one_rejects_weak_score(mock_embedding, mock_pinecone, mock_llm, sample_candidate):
    mock_embedding()
    mock_pinecone([{**sample_candidate, "score": 0.58}])
    mock_llm({"make_model": "Project Salary"})
    result = match_one_asset(
        {"AssetName": "Project Salary"},
        customer_id=1158,
        request_id="weak",
    )
    assert result["MatchFound"] == "N"
    assert result["MakeModelGenerated"] == "Y"
    assert result["master"]["MakeModel"] == "Project Salary"
    assert result["master"]["MakeModelId"] is None
    assert result["master"]["CategoryId"] is None
    assert result["Message"] == "No acceptable master match"


def test_match_one_fallback_index_message(
    mock_embedding, mock_pinecone, mock_llm, sample_candidate
):
    mock_embedding()
    mock_pinecone([sample_candidate])
    mock_llm(
        {
            "candidate_index": 99,
            "make_model": "ignored",
            "make_model_generated": "Y",
        }
    )
    result = match_one_asset(
        {"AssetName": "hdd 8 gb ram"},
        customer_id=1004,
        request_id="t5",
    )
    assert result["MatchFound"] == "Y"
    assert result["Message"] is None
    assert result["master"]["MakeModel"] == "8 gb ram"
    assert result["master"]["MakeModelId"] == 10338
    assert result["MakeModelGenerated"] == "N"


def test_match_one_generates_when_stored_make_model_is_na(
    mock_embedding, mock_pinecone, mock_llm, sample_candidate
):
    mock_embedding()
    row = {**sample_candidate, "makemodel": "N/A", "makemodelid": 1756, "score": 0.82}
    mock_pinecone([row])
    mock_llm({"make_model": "Aluminium step ladder 6 ft"})
    result = match_one_asset(
        {"AssetName": "Aluminium step ladder 6 ft"},
        customer_id=1158,
        request_id="na",
    )
    assert result["MatchFound"] == "Y"
    assert result["MakeModelGenerated"] == "Y"
    assert result["master"]["MakeModel"] == "Aluminium step ladder 6 ft"
    assert result["master"]["MakeModelId"] is None
    assert result["master"]["CategoryId"] == 1039
    assert result["master"]["SubCategoryId"] == 10067


def test_match_one_master_follows_top_score(
    mock_embedding, mock_pinecone, sample_candidate, monkeypatch
):
    mock_embedding()
    low = {**sample_candidate, "makemodel": "Other model", "makemodelid": 2, "score": 0.70}
    high = {**sample_candidate, "makemodel": "Linear Workstation Small", "makemodelid": 1003, "score": 0.91}
    mock_pinecone([low, high])

    def _boom(*_a, **_k):
        raise AssertionError("llm should not run for a real make model")

    monkeypatch.setattr("asset_master_match.services.match_service.generate_json", _boom)
    result = match_one_asset(
        {"AssetName": "Fixed LINEAR WORKSTATION - 1200x650"},
        customer_id=1158,
        request_id="stable",
    )
    assert result["MakeModelGenerated"] == "N"
    assert result["master"]["MakeModel"] == "Linear Workstation Small"
    assert result["master"]["MakeModelId"] == 1003


def test_batch_partial_failure(
    make_http_request, mock_embedding, mock_pinecone, mock_llm, sample_candidate
):
    mock_embedding()
    mock_pinecone([sample_candidate])
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "N",
        }
    )
    resp = run_asset_master_match(
        make_http_request(
            {
                "CustomerId": 1004,
                "assets": [
                    {"AssetName": "hdd 8 gb ram", "Count": 1},
                    {"AssetType": "PP&E", "Count": 1},  # missing AssetName
                ],
            }
        )
    )
    assert resp.status_code == 200
    body = _body(resp)
    assert "CustomerId" not in body
    assert len(body["results"]) == 2
    assert body["results"][0]["CustomerId"] == 1004
    assert body["results"][0]["MatchFound"] == "Y"
    assert body["results"][0]["Message"] is None
    assert body["results"][1]["MatchFound"] == "N"
    assert body["results"][1].get("Error") is True


def test_multi_customer_requests_flat_results_preserve_customer_scope(
    make_http_request, mock_embedding, mock_llm, sample_candidate, monkeypatch
):
    """Each asset is matched with its group CustomerId (never mixed)."""
    seen_customers: list[int] = []

    def _query(embedding, *, customer_id, category_id=None, asset_class_id=None, request_id=""):
        seen_customers.append(customer_id)
        row = dict(sample_candidate)
        row["customerid"] = customer_id
        return [row]

    mock_embedding()
    monkeypatch.setattr(
        "asset_master_match.services.match_service.query_masterdata",
        _query,
    )
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "N",
        }
    )
    resp = run_asset_master_match(
        make_http_request(
            {
                "requests": [
                    {
                        "CustomerId": 1001,
                        "assets": [
                            {"AssetName": "laptop one"},
                            {"AssetName": "laptop two"},
                        ],
                    },
                    {
                        "CustomerId": 1004,
                        "assets": [
                            {"AssetName": "pump one"},
                            {"AssetName": "pump two"},
                        ],
                    },
                ]
            }
        )
    )
    assert resp.status_code == 200
    body = _body(resp)
    assert len(body["results"]) == 4
    assert [r["CustomerId"] for r in body["results"]] == [1001, 1001, 1004, 1004]
    # Parallel workers may finish out of order; each call must still use the job's customer.
    assert sorted(seen_customers) == [1001, 1001, 1004, 1004]
    assert all(r["MatchFound"] == "Y" for r in body["results"])
    assert all(r["master"]["MakeModelId"] == 10338 for r in body["results"])
    assert body["request_id"]


def test_success_drops_caller_error_flag(mock_embedding, mock_pinecone, mock_llm, sample_candidate):
    mock_embedding()
    mock_pinecone([sample_candidate])
    mock_llm(
        {
            "candidate_index": 1,
            "make_model": "8 gb ram",
            "make_model_generated": "N",
        }
    )
    result = match_one_asset(
        {
            "AssetName": "hdd 8 gb ram",
            "Error": True,
            "Message": "from caller",
            "MatchFound": "N",
            "master": {"MakeModel": "injected"},
        },
        customer_id=1004,
        request_id="echo",
    )
    assert result["MatchFound"] == "Y"
    assert result["Message"] is None
    assert "Error" not in result
    assert result["master"]["MakeModel"] == "8 gb ram"


def test_past_deadline_skips_embed(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("embed should not run after the deadline")

    monkeypatch.setattr(match_service, "generate_embedding", _boom)
    result = match_one_asset(
        {"AssetName": "pump"},
        customer_id=1004,
        request_id="dead",
        deadline=time.monotonic() - 1,
    )
    assert result["Message"] == MSG_TIMED_OUT
    assert result["Error"] is True
    assert result["MatchFound"] == "N"


def test_batch_deadline_returns_finished_rows(make_http_request, monkeypatch):
    monkeypatch.setattr(match_service, "BATCH_DEADLINE_SEC", 0.4)

    def _match(asset, *, customer_id, request_id, deadline=None):
        if asset["AssetName"] == "slow":
            time.sleep(2)
        return {
            "CustomerId": customer_id,
            "AssetName": asset["AssetName"],
            "master": {"MakeModel": "kept"},
            "MatchFound": "Y",
            "MakeModelGenerated": "N",
            "Message": None,
        }

    monkeypatch.setattr(match_service, "match_one_asset", _match)
    resp = run_asset_master_match(
        make_http_request(
            {
                "CustomerId": 1004,
                "assets": [
                    {"AssetName": "fast"},
                    {"AssetName": "slow"},
                ],
            }
        )
    )
    assert resp.status_code == 200
    body = _body(resp)
    by_name = {row["AssetName"]: row for row in body["results"]}
    assert by_name["fast"]["MatchFound"] == "Y"
    assert by_name["slow"]["Message"] == MSG_TIMED_OUT
    assert by_name["slow"]["Error"] is True


def test_http_503_when_every_row_is_upstream(make_http_request, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(match_service, "generate_embedding", _boom)
    resp = run_asset_master_match(
        make_http_request(
            {
                "CustomerId": 1004,
                "assets": [{"AssetName": "pump"}, {"AssetName": "laptop"}],
            }
        )
    )
    assert resp.status_code == 503
    body = _body(resp)
    assert body["request_id"]
    assert [row["Message"] for row in body["results"]] == [
        "Embedding service unavailable",
        "Embedding service unavailable",
    ]


def test_http_200_when_no_candidates(make_http_request, mock_embedding, mock_pinecone, mock_llm):
    mock_embedding()
    mock_pinecone([])
    mock_llm({"make_model": "unknown"})
    resp = run_asset_master_match(
        make_http_request({"CustomerId": 1004, "assets": [{"AssetName": "unknown"}]})
    )
    assert resp.status_code == 200
    body = _body(resp)
    assert body["request_id"]
    assert body["results"][0]["Message"] == "No master data candidates found"
    assert "Error" not in body["results"][0]
