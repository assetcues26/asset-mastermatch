"""Pinecone filtered vector search for master data."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

from pinecone import Pinecone

from asset_master_match.core.config import (
    PINECONE_INDEX_NAME,
    PINECONE_MAX_CALLS_PER_DAY,
    PINECONE_MAX_CALLS_PER_HOUR,
    PINECONE_MAX_CALLS_PER_MINUTE,
    PINECONE_TOP_K,
)
from asset_master_match.utils.helpers import to_int_or_none

_index = None
_index_lock = threading.Lock()


class PineconeRateLimiter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.minute_calls = 0
        self.hourly_calls = 0
        self.daily_calls = 0
        self.last_reset_minute = None
        self.last_reset_hour = None
        self.last_reset_day = None

    def check_and_record(self) -> None:
        with self.lock:
            now = datetime.now()
            if self.last_reset_day is None:
                self.last_reset_day = now.day
                self.last_reset_hour = now.hour
                self.last_reset_minute = now.minute
            if now.day != self.last_reset_day:
                self.daily_calls = 0
                self.last_reset_day = now.day
            if now.hour != self.last_reset_hour:
                self.hourly_calls = 0
                self.last_reset_hour = now.hour
            if now.minute != self.last_reset_minute:
                self.minute_calls = 0
                self.last_reset_minute = now.minute
            if self.minute_calls >= PINECONE_MAX_CALLS_PER_MINUTE:
                raise RuntimeError(
                    f"Pinecone rate limit: {PINECONE_MAX_CALLS_PER_MINUTE}/min exceeded"
                )
            if self.hourly_calls >= PINECONE_MAX_CALLS_PER_HOUR:
                raise RuntimeError(
                    f"Pinecone rate limit: {PINECONE_MAX_CALLS_PER_HOUR}/hour exceeded"
                )
            if self.daily_calls >= PINECONE_MAX_CALLS_PER_DAY:
                raise RuntimeError(
                    f"Pinecone rate limit: {PINECONE_MAX_CALLS_PER_DAY}/day exceeded"
                )
            self.minute_calls += 1
            self.hourly_calls += 1
            self.daily_calls += 1


_rate_limiter = PineconeRateLimiter()


def get_pinecone_index():
    global _index
    with _index_lock:
        if _index is None:
            api_key = os.getenv("PINECONE_API_KEY")
            index_name = PINECONE_INDEX_NAME or os.getenv("PINECONE_INDEX_NAME", "")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            if not index_name:
                raise ValueError("PINECONE_INDEX_NAME environment variable not set")
            pc = Pinecone(api_key=api_key)
            _index = pc.Index(index_name)
            logging.info("Pinecone index '%s' connected", index_name)
        return _index


def build_metadata_filter(
    customer_id: int,
    category_id: int | None = None,
    asset_class_id: int | None = None,
) -> dict[str, Any]:
    """Always filter customerid; add categoryid / assetclassid only when provided.

    tenantid is intentionally ignored.
    """
    clauses: list[dict[str, Any]] = [{"customerid": {"$eq": customer_id}}]
    if category_id is not None:
        clauses.append({"categoryid": {"$eq": category_id}})
    if asset_class_id is not None:
        clauses.append({"assetclassid": {"$eq": asset_class_id}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _match_as_dict(match: Any) -> dict[str, Any]:
    if isinstance(match, dict):
        return match
    if hasattr(match, "to_dict"):
        return match.to_dict()
    return {
        "id": getattr(match, "id", None),
        "score": getattr(match, "score", None),
        "metadata": getattr(match, "metadata", None) or {},
    }


def _row_from_match(match: Any) -> dict[str, Any]:
    data = _match_as_dict(match)
    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = dict(meta) if meta else {}
    row = dict(meta)
    row["_id"] = data.get("id")
    row["score"] = data.get("score")
    return row


def query_masterdata(
    embedding: list[float],
    *,
    customer_id: int,
    category_id: int | None = None,
    asset_class_id: int | None = None,
    top_k: int | None = None,
    request_id: str = "",
) -> list[dict[str, Any]]:
    if not embedding:
        return []

    _rate_limiter.check_and_record()
    metadata_filter = build_metadata_filter(customer_id, category_id, asset_class_id)
    index = get_pinecone_index()
    k = top_k if top_k is not None else PINECONE_TOP_K

    start = time.time()
    results = index.query(
        vector=embedding,
        top_k=k,
        filter=metadata_filter,
        include_metadata=True,
    )
    elapsed_ms = (time.time() - start) * 1000

    if isinstance(results, dict):
        matches = results.get("matches", []) or []
    else:
        matches = getattr(results, "matches", None) or []
    candidates = [_row_from_match(m) for m in matches]

    # Defense in depth: drop anything outside requested scope if index misbehaves.
    kept: list[dict[str, Any]] = []
    for row in candidates:
        if to_int_or_none(row.get("customerid")) != customer_id:
            continue
        if category_id is not None and to_int_or_none(row.get("categoryid")) != category_id:
            continue
        if asset_class_id is not None and to_int_or_none(row.get("assetclassid")) != asset_class_id:
            continue
        kept.append(row)

    if len(kept) != len(candidates):
        logging.warning(
            "[%s] Dropped %s out-of-scope Pinecone row(s)",
            request_id,
            len(candidates) - len(kept),
        )

    logging.info(
        "[%s] Pinecone query %.0fms filter=%s hits=%s",
        request_id,
        elapsed_ms,
        metadata_filter,
        len(kept),
    )
    return kept
