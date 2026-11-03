"""HTTP + orchestration tests with Gemini/Pinecone mocked."""

from __future__ import annotations

import json

from asset_master_match.core.config import MAX_ASSET_NAME_LENGTH
from asset_master_match.services.match_service import match_one_asset, run_asset_master_match


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


def test_match_one_no_pinecone_hits(mock_embedding, mock_pinecone, sample_candidate):
    mock_embedding()
    mock_pinecone([])
    result = match_one_asset(
        {"AssetName": "unknown device"},
        customer_id=1004,
        request_id="t2",
    )
    assert result["MatchFound"] == "N"
    assert result["master"] is None
    assert result["MakeModelGenerated"] == "N"
    assert "No master data" in result["Message"]


def test_match_one_success_no_llm_reason(
    mock_embedding, mock_pinecone, mock_llm, sample_candidate
):
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
    assert result["MakeModelGenerated"] == "Y"
    assert result["CustomerId"] == 1004
    assert result["Cost"] == 50000
    assert result["master"]["MakeModel"] == "16 gb ram"
    assert result["master"]["MakeModelId"] is None
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
    assert len(seen) == 1
    assert seen[0] == "hdd 16 gb ram\nDDR4 module for rack server"


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


def test_match_one_select_error_hides_exception(
    mock_embedding, mock_pinecone, monkeypatch
):
    mock_embedding()
    mock_pinecone([{"customerid": 1004, "makemodel": "x"}])

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
        raise RuntimeError(
            "503 UNAVAILABLE. This model is currently experiencing high demand."
        )

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


def test_match_one_select_unavailable(
    mock_embedding, mock_pinecone, monkeypatch
):
    mock_embedding()
    mock_pinecone([{"customerid": 1004, "makemodel": "x"}])

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


def test_match_one_select_rate_limited(
    mock_embedding, mock_pinecone, monkeypatch
):
    from asset_master_match.services.gemini_service import RateLimitExceeded

    mock_embedding()
    mock_pinecone([{"customerid": 1004, "makemodel": "x"}])

    def _boom(*_a, **_k):
        raise RateLimitExceeded("GEMINI per-minute call limit exceeded")

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
    assert result["Message"] == "fallback_index"
    assert result["master"]["MakeModel"] == "8 gb ram"
    assert result["MakeModelGenerated"] == "N"


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
