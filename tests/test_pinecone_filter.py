"""Pinecone metadata filter rules."""

from asset_master_match.services import pinecone_service
from asset_master_match.services.pinecone_service import build_metadata_filter, query_masterdata


def test_filter_customer_only():
    assert build_metadata_filter(1004) == {"customerid": {"$eq": 1004}}


def test_filter_customer_and_category():
    assert build_metadata_filter(1004, category_id=1039) == {
        "$and": [
            {"customerid": {"$eq": 1004}},
            {"categoryid": {"$eq": 1039}},
        ]
    }


def test_filter_customer_and_asset_class():
    assert build_metadata_filter(1004, asset_class_id=1107) == {
        "$and": [
            {"customerid": {"$eq": 1004}},
            {"assetclassid": {"$eq": 1107}},
        ]
    }


def test_filter_all_three():
    assert build_metadata_filter(1004, category_id=1039, asset_class_id=1107) == {
        "$and": [
            {"customerid": {"$eq": 1004}},
            {"categoryid": {"$eq": 1039}},
            {"assetclassid": {"$eq": 1107}},
        ]
    }


def test_filter_never_includes_tenantid():
    filt = build_metadata_filter(1004, category_id=1, asset_class_id=2)
    blob = str(filt)
    assert "tenantid" not in blob


def test_query_passes_timeout(monkeypatch):
    seen: dict = {}

    class _Index:
        def query(self, **kwargs):
            seen.update(kwargs)
            return {"matches": []}

    monkeypatch.setattr(pinecone_service, "get_pinecone_index", lambda: _Index())
    monkeypatch.setattr(pinecone_service, "PINECONE_CALL_TIMEOUT_SEC", 10)
    assert query_masterdata([0.1, 0.2], customer_id=1004, request_id="t") == []
    assert seen["timeout"] == 10
    assert seen["top_k"] == pinecone_service.PINECONE_TOP_K
