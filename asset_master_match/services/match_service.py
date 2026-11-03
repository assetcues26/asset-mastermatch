"""Orchestrate embed → Pinecone → LLM select for one or many assets."""

from __future__ import annotations

import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import azure.functions as func

from asset_master_match.core.config import (
    MAX_ASSET_DESCRIPTION_LENGTH,
    MAX_ASSET_NAME_LENGTH,
    MAX_ASSETS_PER_REQUEST,
    MAX_EMBED_QUERY_LENGTH,
    MAX_MAKE_MODEL_LENGTH,
    MAX_PARALLEL_ITEMS,
)
from asset_master_match.prompts.builders import build_select_prompt
from asset_master_match.services.gemini_service import (
    RateLimitExceeded,
    generate_embedding,
    generate_json,
)
from asset_master_match.services.pinecone_service import query_masterdata
from asset_master_match.utils.helpers import (
    build_embed_query,
    normalize_text,
    safe_str,
    sanitize_make_model,
    texts_match,
    to_int_or_none,
)

INPUT_ECHO_KEYS = (
    "AcquisitionDate",
    "AssetNumber",
    "SubNumber",
    "AssetName",
    "AssetDescription",
    "AssetType",
    "PlantLocation",
    "AssetClass",
    "AssetClassId",
    "Category",
    "CategoryId",
    "Cost",
    "Count",
)

MSG_FALLBACK_INDEX = "fallback_index"
MSG_ASSET_NAME_REQUIRED = "AssetName is required"
MSG_ASSET_NAME_TOO_LONG = "AssetName exceeds maximum length"
MSG_NO_CANDIDATES = "No master data candidates found"
MSG_EMBED_FAILED = "Failed to embed AssetName"
MSG_EMBED_UNAVAILABLE = "Embedding service unavailable"
MSG_EMBED_ERROR = "Embedding service error"
MSG_PINECONE_UNAVAILABLE = "Master data search unavailable"
MSG_SELECT_UNAVAILABLE = "Gemini selection unavailable"
MSG_SELECT_ERROR = "Failed to select master match"
MSG_SELECT_RATE_LIMITED = "Gemini rate limit hit"
MSG_PIPELINE_ERROR = "Match pipeline error"


def _log_stage_failure(item_id: str, stage: str, exc: BaseException) -> None:
    """Log stage failures so they show in ``func start`` (incl. worker threads)."""
    logging.exception("[%s] %s failed: %s", item_id, stage, exc)
    # Azure Functions often drops worker-thread logging; stderr reaches the host console.
    print(
        f"[{item_id}] {stage} failed: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _is_unavailable_exc(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "503",
            "unavailable",
            "high demand",
            "overloaded",
            "service unavailable",
        )
    )


def _is_rate_limit_exc(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitExceeded):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "429",
            "resource_exhausted",
            "rate limit",
            "rate_limit",
            "quota",
        )
    )


def _message_for_embed_exc(exc: BaseException) -> str:
    if _is_unavailable_exc(exc):
        return MSG_EMBED_UNAVAILABLE
    return MSG_EMBED_ERROR


def _message_for_select_exc(exc: BaseException) -> str:
    if _is_rate_limit_exc(exc):
        return MSG_SELECT_RATE_LIMITED
    if _is_unavailable_exc(exc):
        return MSG_SELECT_UNAVAILABLE
    return MSG_SELECT_ERROR


def _json_response(body: dict[str, Any], status: int = 200) -> func.HttpResponse:
    import json

    return func.HttpResponse(
        json.dumps(body, ensure_ascii=False),
        status_code=status,
        mimetype="application/json",
    )


def _echo_input(asset: dict[str, Any], customer_id: int) -> dict[str, Any]:
    echoed: dict[str, Any] = {"CustomerId": customer_id}
    for key in INPUT_ECHO_KEYS:
        if key in asset:
            echoed[key] = asset[key]
    # Preserve any extra caller fields without inventing names.
    for key, value in asset.items():
        if key not in echoed and key not in ("CustomerId",):
            echoed[key] = value
    return echoed


def _master_from_candidate(
    candidate: dict[str, Any],
    *,
    make_model: str,
    make_model_generated: str,
) -> dict[str, Any]:
    generated = make_model_generated.upper() == "Y"
    make_model_id = None if generated else to_int_or_none(candidate.get("makemodelid"))
    return {
        "AssetClass": safe_str(candidate.get("assetclass")),
        "AssetClassId": to_int_or_none(candidate.get("assetclassid")),
        "Category": safe_str(candidate.get("category") or candidate.get("categoryname")),
        "CategoryId": to_int_or_none(candidate.get("categoryid")),
        "SubCategory": safe_str(candidate.get("subcategory")),
        "SubCategoryId": to_int_or_none(candidate.get("subcategoryid")),
        "MakeModel": make_model,
        "MakeModelId": make_model_id,
    }


def _no_match_result(echoed: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        **echoed,
        "master": None,
        "MatchFound": "N",
        "MakeModelGenerated": "N",
        "Message": message,
    }


def _error_result(echoed: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        **echoed,
        "master": None,
        "MatchFound": "N",
        "MakeModelGenerated": "N",
        "Message": message,
        "Error": True,
    }


def _apply_llm_choice(
    asset_name: str,
    candidates: list[dict[str, Any]],
    llm: dict[str, Any],
) -> tuple[dict[str, Any], str, str | None]:
    """Return (master, make_model_generated, message).

    ``message`` is a fixed status string from this service (never LLM prose).
    Classification always comes from the chosen candidate.
    """
    index = to_int_or_none(llm.get("candidate_index"))
    if index is None or index < 1 or index > len(candidates):
        # Fallback: first candidate; keep its MakeModel.
        chosen = candidates[0]
        mm = safe_str(chosen.get("makemodel")) or ""
        return (
            _master_from_candidate(chosen, make_model=mm, make_model_generated="N"),
            "N",
            MSG_FALLBACK_INDEX,
        )

    chosen = candidates[index - 1]
    pinecone_mm = safe_str(chosen.get("makemodel")) or ""
    proposed = safe_str(llm.get("make_model")) or pinecone_mm
    flag_raw = str(llm.get("make_model_generated") or "N").strip().upper()
    generated = flag_raw == "Y"

    # Enforce: if text equals Pinecone MakeModel, never mark generated / never null the id.
    if texts_match(proposed, pinecone_mm):
        proposed = pinecone_mm
        generated = False
    # If model claimed N but changed text, treat as generated.
    if not generated and not texts_match(proposed, pinecone_mm):
        generated = True
    # Empty proposed → keep Pinecone value.
    if not proposed:
        proposed = pinecone_mm
        generated = False

    # Generated MakeModel is untrusted — sanitize length/control chars.
    if generated:
        cleaned = sanitize_make_model(proposed, MAX_MAKE_MODEL_LENGTH)
        if not cleaned:
            proposed = pinecone_mm
            generated = False
        else:
            proposed = cleaned
            if texts_match(proposed, pinecone_mm):
                proposed = pinecone_mm
                generated = False

    flag = "Y" if generated else "N"
    master = _master_from_candidate(
        chosen,
        make_model=proposed,
        make_model_generated=flag,
    )
    logging.debug(
        "LLM pick asset=%r index=%s generated=%s pinecone_mm=%r final_mm=%r",
        asset_name,
        index,
        flag,
        pinecone_mm,
        proposed,
    )
    return master, flag, None


def match_one_asset(
    asset: dict[str, Any],
    *,
    customer_id: int,
    request_id: str,
) -> dict[str, Any]:
    echoed = _echo_input(asset, customer_id)
    asset_name = normalize_text(asset.get("AssetName"))
    if not asset_name:
        return _error_result(echoed, MSG_ASSET_NAME_REQUIRED)
    if len(asset_name) > MAX_ASSET_NAME_LENGTH:
        return _error_result(echoed, MSG_ASSET_NAME_TOO_LONG)

    # Optional; empty/whitespace treated as absent. Overlong text is truncated in
    # build_embed_query / prompt builders (do not fail the item).
    asset_description = normalize_text(asset.get("AssetDescription"))

    category_id = to_int_or_none(asset.get("CategoryId"))
    asset_class_id = to_int_or_none(asset.get("AssetClassId"))
    item_id = f"{request_id}:{asset_name[:40]}"

    query_text = build_embed_query(
        asset_name,
        asset_description or None,
        max_description_length=MAX_ASSET_DESCRIPTION_LENGTH,
        max_embed_query_length=MAX_EMBED_QUERY_LENGTH,
    )

    try:
        embedding = generate_embedding(query_text, request_id=item_id)
    except Exception as exc:
        _log_stage_failure(item_id, "embed", exc)
        return _error_result(echoed, _message_for_embed_exc(exc))
    if not embedding:
        return _error_result(echoed, MSG_EMBED_FAILED)

    try:
        candidates = query_masterdata(
            embedding,
            customer_id=customer_id,
            category_id=category_id,
            asset_class_id=asset_class_id,
            request_id=item_id,
        )
    except Exception as exc:
        _log_stage_failure(item_id, "pinecone", exc)
        return _error_result(echoed, MSG_PINECONE_UNAVAILABLE)
    if not candidates:
        return _no_match_result(echoed, MSG_NO_CANDIDATES)

    try:
        prompt = build_select_prompt(
            asset_name,
            candidates,
            asset_description=asset_description,
        )
        llm = generate_json(prompt, request_id=item_id)
        master, generated, message = _apply_llm_choice(asset_name, candidates, llm)
        return {
            **echoed,
            "master": master,
            "MatchFound": "Y",
            "MakeModelGenerated": generated,
            "Message": message,
        }
    except Exception as exc:
        _log_stage_failure(item_id, "select", exc)
        return _error_result(echoed, _message_for_select_exc(exc))


def _parse_body(req: func.HttpRequest) -> dict[str, Any]:
    try:
        body = req.get_json()
    except ValueError as exc:
        raise ValueError("Request body must be JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return body


def _normalize_work_items(
    body: dict[str, Any],
) -> tuple[list[tuple[int, dict[str, Any]]], str | None]:
    """Build ordered (customer_id, asset) jobs from Option A ``requests`` or legacy shape.

    Returns (work_items, error_message). error_message set => HTTP 400.
    Each job keeps its own customer_id so Pinecone never mixes customers.
    """
    work: list[tuple[int, dict[str, Any]]] = []

    if "requests" in body and body.get("requests") is not None:
        groups = body.get("requests")
        if not isinstance(groups, list) or not groups:
            return [], "requests must be a non-empty array"

        for g_idx, group in enumerate(groups):
            if not isinstance(group, dict):
                return [], f"requests[{g_idx}] must be an object"
            customer_id = to_int_or_none(group.get("CustomerId"))
            if customer_id is None:
                return [], f"requests[{g_idx}].CustomerId is required"
            assets = group.get("assets")
            if not isinstance(assets, list) or not assets:
                return [], f"requests[{g_idx}].assets must be a non-empty array"
            for a_idx, asset in enumerate(assets):
                if not isinstance(asset, dict):
                    return [], f"requests[{g_idx}].assets[{a_idx}] must be an object"
                work.append((customer_id, asset))
    else:
        # Legacy: single CustomerId + assets (same isolation; one customer for all rows).
        customer_id = to_int_or_none(body.get("CustomerId"))
        if customer_id is None:
            return [], "CustomerId is required (or send requests[])"
        assets = body.get("assets")
        if not isinstance(assets, list) or not assets:
            return [], "assets must be a non-empty array"
        for a_idx, asset in enumerate(assets):
            if not isinstance(asset, dict):
                return [], f"assets[{a_idx}] must be an object"
            work.append((customer_id, asset))

    if len(work) > MAX_ASSETS_PER_REQUEST:
        return [], f"At most {MAX_ASSETS_PER_REQUEST} assets allowed per request"

    return work, None


def _run_work_items(
    work: list[tuple[int, dict[str, Any]]],
    *,
    request_id: str,
) -> list[dict[str, Any]]:
    """Match each asset with its own customer_id; preserve input order in flat results."""
    if len(work) == 1:
        customer_id, asset = work[0]
        return [
            match_one_asset(
                asset,
                customer_id=customer_id,
                request_id=request_id,
            )
        ]

    results_slot: list[dict[str, Any] | None] = [None] * len(work)
    workers = max(1, min(MAX_PARALLEL_ITEMS, len(work)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                match_one_asset,
                asset,
                customer_id=customer_id,
                request_id=request_id,
            ): idx
            for idx, (customer_id, asset) in enumerate(work)
        }
        for future in as_completed(futures):
            idx = futures[future]
            customer_id, asset = work[idx]
            try:
                results_slot[idx] = future.result()
            except Exception as exc:
                _log_stage_failure(f"{request_id}:worker[{idx}]", "pipeline", exc)
                results_slot[idx] = _error_result(
                    _echo_input(asset, customer_id),
                    MSG_PIPELINE_ERROR,
                )
    return [item for item in results_slot if item is not None]


def run_asset_master_match(req: func.HttpRequest) -> func.HttpResponse:
    request_id = str(uuid.uuid4())[:8]
    logging.info("[%s] asset_master_match received", request_id)

    try:
        body = _parse_body(req)
    except ValueError as exc:
        return _json_response({"error": str(exc)}, status=400)

    work, err = _normalize_work_items(body)
    if err:
        return _json_response({"error": err}, status=400)

    results = _run_work_items(work, request_id=request_id)

    # Main-thread summary for Postman/debug (worker logs can be dropped by the host).
    for idx, item in enumerate(results):
        if item and item.get("Error"):
            logging.error(
                "[%s] assets[%s] Error Message=%s CustomerId=%s AssetName=%r",
                request_id,
                idx,
                item.get("Message"),
                item.get("CustomerId"),
                item.get("AssetName"),
            )

    # Flat results only — each row echoes its own CustomerId (no mixed top-level id).
    return _json_response({"results": results})

