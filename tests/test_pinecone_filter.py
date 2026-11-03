"""Pinecone metadata filter rules."""

from asset_master_match.services.pinecone_service import build_metadata_filter


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
