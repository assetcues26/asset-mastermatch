"""Orchestrate embed → Pinecone → LLM select for one or many assets."""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import azure.functions as func

from asset_master_match.core.config import (
    BATCH_DEADLINE_SEC,
    EMBED_DESCRIPTION_MAX_LENGTH,
    ENABLE_LLM_REVIEW,
    ENABLE_MATCH_DEBUG_LOG,
    ENABLE_SUBCATEGORY_GENERATION,
    MAX_ASSET_DESCRIPTION_LENGTH,
    MAX_ASSET_NAME_LENGTH,
    MAX_ASSETS_PER_REQUEST,
    MAX_EMBED_QUERY_LENGTH,
    MAX_MAKE_MODEL_LENGTH,
    MAX_PARALLEL_ITEMS,
    MAX_PROMPT_FIELD_LENGTH,
    MIN_MATCH_SCORE,
    PINECONE_TOP_K,
    USE_COMBINED_EMBED_QUERY,
)
from asset_master_match.prompts.builders import (
    build_select_candidate_prompt,
    build_subcategory_prompt,
)
from asset_master_match.services.gemini_service import (
    generate_embedding,
    generate_json,
)
from asset_master_match.services.pinecone_service import query_masterdata
from asset_master_match.utils.debug_match_log import maybe_log_match
from asset_master_match.utils.helpers import (
    build_embed_query,
    clean_description_for_embed,
    normalize_text,
    safe_str,
    sanitize_make_model,
    sanitize_prompt_field,
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
MSG_NO_ACCEPTABLE_MATCH = "No acceptable master match"
MSG_EMBED_FAILED = "Failed to embed AssetName"
MSG_EMBED_UNAVAILABLE = "Embedding service unavailable"
MSG_EMBED_ERROR = "Embedding service error"
MSG_PINECONE_UNAVAILABLE = "Master data search unavailable"
MSG_SELECT_UNAVAILABLE = "Gemini selection unavailable"
MSG_SELECT_ERROR = "Failed to select master match"
MSG_SELECT_RATE_LIMITED = "Gemini rate limit hit"
MSG_PIPELINE_ERROR = "Match pipeline error"
MSG_TIMED_OUT = "Match timed out"

# Caller fields with these names are not copied onto the result. This service sets them.
_RESERVED_RESULT_KEYS = frozenset(
    {
        "CustomerId",
        "master",
        "MatchFound",
        "MakeModelGenerated",
        "SubCategoryGenerated",
        "Message",
        "Error",
    }
)

_TOKEN = re.compile(r"[a-z0-9]+")
_INVOICE = re.compile(r"\b(inv|invoice|charges)\b|\d{6,}", re.IGNORECASE)
_MIN_TOKEN_LEN = 4
_MIN_SNAP_LEN = 8


def _score(row: dict[str, Any]) -> float | None:
    try:
        return float(row.get("score"))
    except (TypeError, ValueError):
        return None


def _best_score(rows: list[dict[str, Any]]) -> float | None:
    scores = [score for row in rows if (score := _score(row)) is not None]
    return max(scores) if scores else None


def _tokens(text: Any) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(normalize_text(text).casefold())
        if len(token) >= _MIN_TOKEN_LEN
    ]


def _fit(asset_name: str, make_model: Any) -> tuple[int, int]:
    """(matched tokens, negative unmatched) so a closer name sorts first."""
    tokens = _tokens(make_model)
    if not tokens:
        return (0, 0)
    asset = set(_tokens(asset_name))
    matched = sum(1 for token in tokens if token in asset)
    return (matched, -(len(tokens) - matched))


def _shares_long_token(asset_name: str, make_model: Any) -> bool:
    asset = set(_tokens(asset_name))
    return any(token in asset for token in _tokens(make_model))


def _looks_like_invoice(text: str) -> bool:
    return bool(_INVOICE.search(text))


def _merge_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the higher score when name-only and description search return the same row."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for group in groups:
        for row in group:
            key = str(row.get("_id") or row.get("makemodelid") or id(row))
            current = merged.get(key)
            if current is None:
                merged[key] = row
                order.append(key)
                continue
            if (_score(row) or 0) > (_score(current) or 0):
                merged[key] = row
    rows = [merged[key] for key in order]
    rows.sort(key=lambda row: _score(row) or 0, reverse=True)
    return rows[: max(PINECONE_TOP_K, 8)]


def _snap_to_existing(
    proposed: str, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Use the master whose MakeModel is already inside the rewritten text."""
    proposed_norm = normalize_text(proposed).casefold()
    best: dict[str, Any] | None = None
    best_len = 0
    for row in candidates:
        make_model = safe_str(row.get("makemodel")) or ""
        norm = normalize_text(make_model).casefold()
        if len(norm) < _MIN_SNAP_LEN or norm not in proposed_norm:
            continue
        if len(norm) > best_len:
            best = row
            best_len = len(norm)
    return best


# Every row carrying one of these means the upstream call failed, not the input.
_UPSTREAM_MESSAGES = frozenset(
    {
        MSG_EMBED_FAILED,
        MSG_EMBED_UNAVAILABLE,
        MSG_EMBED_ERROR,
        MSG_PINECONE_UNAVAILABLE,
        MSG_SELECT_UNAVAILABLE,
        MSG_SELECT_ERROR,
        MSG_SELECT_RATE_LIMITED,
        MSG_PIPELINE_ERROR,
        MSG_TIMED_OUT,
    }
)


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
        if key not in echoed and key not in _RESERVED_RESULT_KEYS:
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
        "SubCategoryGenerated": "N",
        "Message": message,
    }


def _generated_without_master(
    echoed: dict[str, Any],
    asset_name: str,
    asset_description: str,
    request_id: str,
    message: str,
    deadline: float | None,
) -> dict[str, Any]:
    """No master row fit. Still return a MakeModel so the output is not blank."""
    if _deadline_hit(deadline):
        return _error_result(echoed, MSG_TIMED_OUT)
    proposed = ""
    try:
        proposed = _generate_make_model_text(asset_name, asset_description, request_id)
    except Exception as exc:
        _log_stage_failure(request_id, "select", exc)
    if _is_placeholder_make_model(proposed):
        proposed = sanitize_make_model(asset_name, MAX_MAKE_MODEL_LENGTH)

    sub_category = None
    sub_generated = "N"
    if ENABLE_SUBCATEGORY_GENERATION and not _deadline_hit(deadline):
        try:
            sub_category = _generate_subcategory_text(
                asset_name, asset_description, request_id
            )
        except Exception as exc:
            _log_stage_failure(request_id, "subcategory", exc)
        if sub_category:
            sub_generated = "Y"
        else:
            sub_category = None

    # Echo caller's AssetClass/Category into master when present (text and/or ids).
    # Do not invent values — only copy what the input already had.
    return {
        **echoed,
        "master": {
            "AssetClass": safe_str(echoed.get("AssetClass")),
            "AssetClassId": to_int_or_none(echoed.get("AssetClassId")),
            "Category": safe_str(echoed.get("Category")),
            "CategoryId": to_int_or_none(echoed.get("CategoryId")),
            "SubCategory": sub_category,
            "SubCategoryId": None,
            "MakeModel": proposed,
            "MakeModelId": None,
        },
        "MatchFound": "N",
        "MakeModelGenerated": "Y",
        "SubCategoryGenerated": sub_generated,
        "Message": message,
    }


def _error_result(echoed: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        **echoed,
        "master": None,
        "MatchFound": "N",
        "MakeModelGenerated": "N",
        "SubCategoryGenerated": "N",
        "Message": message,
        "Error": True,
    }


def _apply_llm_choice(
    asset_name: str,
    candidates: list[dict[str, Any]],
    llm: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return (master, make_model_generated, message).

    ``message`` is a fixed status string from this service (never LLM prose).
    Classification always comes from the chosen candidate.
    ``master`` is None when the model reports that nothing fits.
    """
    index = to_int_or_none(llm.get("candidate_index"))
    if index == 0:
        return None, "N", MSG_NO_ACCEPTABLE_MATCH
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

    # "LINEAR WORKSTATION - 1200x650" still is the existing LINEAR WORKSTATION row.
    if generated:
        snapped = _snap_to_existing(proposed, candidates)
        if snapped is not None:
            chosen = snapped
            proposed = safe_str(snapped.get("makemodel")) or proposed
            pinecone_mm = proposed
            generated = False

    # Prefer the candidate whose MakeModel actually shares the asset's model tokens.
    best = max(candidates, key=lambda row: _fit(asset_name, row.get("makemodel")))
    best_fit = _fit(asset_name, best.get("makemodel"))
    chosen_fit = _fit(asset_name, chosen.get("makemodel"))
    if best_fit[0] >= 2 and best_fit > chosen_fit:
        chosen = best
        proposed = safe_str(chosen.get("makemodel")) or ""
        pinecone_mm = proposed
        generated = False

    # A related master already exists. Do not mint a new MakeModel from the asset text.
    if generated and _shares_long_token(asset_name, pinecone_mm):
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


_PLACEHOLDER_MAKE_MODELS = frozenset(
    {"", "na", "none", "null", "nil", "notavailable", "notapplicable"}
)


def _is_placeholder_make_model(value: Any) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", normalize_text(value).casefold())
    return compact in _PLACEHOLDER_MAKE_MODELS


def _stable_master(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest Pinecone score, then lowest MakeModelId. Same input, same row."""

    def sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
        score = _score(row)
        mmid = to_int_or_none(row.get("makemodelid"))
        return (
            -(score if score is not None else -1.0),
            mmid if mmid is not None else 10**12,
            str(row.get("_id") or ""),
        )

    return sorted(candidates, key=sort_key)[0]


def _generate_make_model_text(asset_name: str, asset_description: str, request_id: str) -> str:
    """Fill a blank or N/A MakeModel. Does not choose a different master row."""
    prompt = (
        "The matched master row has no MakeModel (it is N/A). "
        "Write the MakeModel text for this asset. "
        "Use AssetName. Add a concrete size or model from AssetDescription only when it belongs to that product. "
        "Do not answer N/A, NA, or empty. "
        'Return only JSON: {"make_model": "<text>"}\n'
        f"<asset_name>\n{asset_name}\n</asset_name>\n"
        f"<asset_description>\n{asset_description or '(none)'}\n</asset_description>"
    )
    data = generate_json(prompt, request_id=request_id)
    return safe_str(data.get("make_model")) or ""


def _generate_subcategory_text(
    asset_name: str, asset_description: str, request_id: str
) -> str:
    """Wipe-path only: short generic SubCategory text. Never invents ids."""
    prompt = build_subcategory_prompt(asset_name, asset_description)
    data = generate_json(prompt, request_id=request_id)
    return sanitize_prompt_field(
        data.get("sub_category"), MAX_PROMPT_FIELD_LENGTH
    )


def _candidates_above_threshold(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in candidates
        if (score := _score(row)) is not None and score >= MIN_MATCH_SCORE
    ]


def _complete_match_from_candidate(
    echoed: dict[str, Any],
    chosen: dict[str, Any],
    asset_name: str,
    asset_description: str,
    request_id: str,
    deadline: float | None,
    *,
    force_generate_make_model: bool = False,
) -> dict[str, Any]:
    """Keep taxonomy from chosen row; generate MakeModel only when needed."""
    stored = safe_str(chosen.get("makemodel")) or ""
    need_generate = force_generate_make_model or _is_placeholder_make_model(stored)
    if not need_generate:
        return {
            **echoed,
            "master": _master_from_candidate(
                chosen, make_model=stored, make_model_generated="N"
            ),
            "MatchFound": "Y",
            "MakeModelGenerated": "N",
            "SubCategoryGenerated": "N",
            "Message": None,
        }

    if _deadline_hit(deadline):
        return _error_result(echoed, MSG_TIMED_OUT)
    try:
        proposed = _generate_make_model_text(
            asset_name, asset_description, request_id
        )
    except Exception as exc:
        _log_stage_failure(request_id, "select", exc)
        return _error_result(echoed, _message_for_select_exc(exc))
    if _is_placeholder_make_model(proposed):
        proposed = sanitize_make_model(asset_name, MAX_MAKE_MODEL_LENGTH)
    return {
        **echoed,
        "master": _master_from_candidate(
            chosen, make_model=proposed, make_model_generated="Y"
        ),
        "MatchFound": "Y",
        "MakeModelGenerated": "Y",
        "SubCategoryGenerated": "N",
        "Message": None,
    }


def _candidate_debug_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact candidate rows for local review logs only."""
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        out.append(
            {
                "rank": rank,
                "score": _score(row),
                "makemodel": safe_str(row.get("makemodel")),
                "makemodelid": to_int_or_none(row.get("makemodelid")),
                "assetclass": safe_str(row.get("assetclass")),
                "category": safe_str(
                    row.get("category") or row.get("categoryname")
                ),
                "subcategory": safe_str(row.get("subcategory")),
                "_id": safe_str(row.get("_id")),
            }
        )
    return out


def _finish_with_debug(
    result: dict[str, Any],
    debug: dict[str, Any] | None,
) -> dict[str, Any]:
    if debug is not None:
        master = result.get("master") if isinstance(result.get("master"), dict) else None
        debug["final"] = {
            "MatchFound": result.get("MatchFound"),
            "MakeModelGenerated": result.get("MakeModelGenerated"),
            "SubCategoryGenerated": result.get("SubCategoryGenerated"),
            "Message": result.get("Message"),
            "Error": result.get("Error"),
            "MakeModel": None if master is None else master.get("MakeModel"),
            "MakeModelId": None if master is None else master.get("MakeModelId"),
        }
        maybe_log_match(debug)
    return result


def _llm_review_match(
    echoed: dict[str, Any],
    asset_name: str,
    asset_description: str,
    candidates: list[dict[str, Any]],
    request_id: str,
    deadline: float | None,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LLM picks among above-threshold candidates; falls back to top-score on failure."""
    eligible = _candidates_above_threshold(candidates)
    if debug is not None:
        debug["eligible_for_llm"] = _candidate_debug_rows(eligible)
    if not eligible:
        if debug is not None:
            debug["decision_path"] = "llm_review→no_eligible"
        return _generated_without_master(
            echoed,
            asset_name,
            asset_description,
            request_id,
            MSG_NO_ACCEPTABLE_MATCH,
            deadline,
        )

    llm: dict[str, Any] | None = None
    try:
        if _deadline_hit(deadline):
            if debug is not None:
                debug["decision_path"] = "llm_review→timed_out"
            return _error_result(echoed, MSG_TIMED_OUT)
        prompt = build_select_candidate_prompt(
            asset_name, eligible, asset_description
        )
        llm = generate_json(prompt, request_id=request_id)
    except Exception as exc:
        _log_stage_failure(request_id, "llm-review", exc)
        logging.warning(
            "[%s] LLM review failed; falling back to top-score match",
            request_id,
        )
        if debug is not None:
            debug["llm_fallback"] = "top_score"
            debug["decision_path"] = "llm_review→fallback_top_score"
        return _complete_match_from_candidate(
            echoed,
            _stable_master(eligible),
            asset_name,
            asset_description,
            request_id,
            deadline,
        )

    if debug is not None:
        debug["llm_raw"] = {
            "candidate_index": (llm or {}).get("candidate_index"),
            "make_model": (llm or {}).get("make_model"),
            "make_model_generated": (llm or {}).get("make_model_generated"),
        }

    index = to_int_or_none((llm or {}).get("candidate_index"))
    if index == 0:
        if debug is not None:
            debug["decision_path"] = "llm_review→index_0_wipe"
        return _generated_without_master(
            echoed,
            asset_name,
            asset_description,
            request_id,
            MSG_NO_ACCEPTABLE_MATCH,
            deadline,
        )
    if index is None or index < 1 or index > len(eligible):
        logging.warning(
            "[%s] LLM review returned invalid index %s; falling back to top-score",
            request_id,
            index,
        )
        if debug is not None:
            debug["llm_fallback"] = "top_score_invalid_index"
            debug["decision_path"] = "llm_review→fallback_invalid_index"
        return _complete_match_from_candidate(
            echoed,
            _stable_master(eligible),
            asset_name,
            asset_description,
            request_id,
            deadline,
        )

    chosen = eligible[index - 1]
    if debug is not None:
        debug["llm_chosen"] = _candidate_debug_rows([chosen])[0]
        debug["decision_path"] = "llm_review→pick"
    flag_raw = str((llm or {}).get("make_model_generated") or "N").strip().upper()
    force_generate = flag_raw == "Y"
    return _complete_match_from_candidate(
        echoed,
        chosen,
        asset_name,
        asset_description,
        request_id,
        deadline,
        force_generate_make_model=force_generate,
    )


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def match_one_asset(
    asset: dict[str, Any],
    *,
    customer_id: int,
    request_id: str,
    deadline: float | None = None,
) -> dict[str, Any]:
    echoed = _echo_input(asset, customer_id)
    asset_name = normalize_text(asset.get("AssetName"))
    if not asset_name:
        return _error_result(echoed, MSG_ASSET_NAME_REQUIRED)
    if len(asset_name) > MAX_ASSET_NAME_LENGTH:
        return _error_result(echoed, MSG_ASSET_NAME_TOO_LONG)
    if _deadline_hit(deadline):
        return _error_result(echoed, MSG_TIMED_OUT)

    # Optional; empty/whitespace treated as absent. Overlong text is truncated in
    # build_embed_query / prompt builders (do not fail the item).
    asset_description = normalize_text(asset.get("AssetDescription"))

    category_id = to_int_or_none(asset.get("CategoryId"))
    asset_class_id = to_int_or_none(asset.get("AssetClassId"))
    item_id = f"{request_id}:{asset_name[:40]}"
    debug: dict[str, Any] | None = None
    if ENABLE_MATCH_DEBUG_LOG:
        debug = {
            "item_id": item_id,
            "CustomerId": customer_id,
            "CategoryId": category_id,
            "AssetClassId": asset_class_id,
            "AssetName": asset_name,
            "AssetDescription_raw": asset_description or None,
            "search_mode": (
                "combined" if USE_COMBINED_EMBED_QUERY else "two_step"
            ),
            "USE_COMBINED_EMBED_QUERY": USE_COMBINED_EMBED_QUERY,
            "ENABLE_LLM_REVIEW": ENABLE_LLM_REVIEW,
            "embed_queries": [],
        }

    if _deadline_hit(deadline):
        return _finish_with_debug(_error_result(echoed, MSG_TIMED_OUT), debug)

    candidates: list[dict[str, Any]] = []
    if USE_COMBINED_EMBED_QUERY:
        # Optional path: one embed of name + cleaned/truncated description.
        cleaned_desc = clean_description_for_embed(
            asset_description,
            max_length=EMBED_DESCRIPTION_MAX_LENGTH,
        )
        query_text = build_embed_query(
            asset_name,
            cleaned_desc or None,
            max_description_length=EMBED_DESCRIPTION_MAX_LENGTH,
            max_embed_query_length=MAX_EMBED_QUERY_LENGTH,
        )
        if debug is not None:
            debug["AssetDescription_cleaned"] = cleaned_desc or None
            debug["AssetDescription_for_embed"] = cleaned_desc or None
            debug["embed_queries"].append(
                {"stage": "combined", "text": query_text}
            )
        try:
            embedding = generate_embedding(query_text, request_id=item_id)
        except Exception as exc:
            _log_stage_failure(item_id, "embed", exc)
            return _finish_with_debug(
                _error_result(echoed, _message_for_embed_exc(exc)), debug
            )
        if not embedding:
            return _finish_with_debug(
                _error_result(echoed, MSG_EMBED_FAILED), debug
            )
        if _deadline_hit(deadline):
            return _finish_with_debug(
                _error_result(echoed, MSG_TIMED_OUT), debug
            )
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
            return _finish_with_debug(
                _error_result(echoed, MSG_PINECONE_UNAVAILABLE), debug
            )
    else:
        # Default: name-first embed, optional second query with raw description.
        try:
            name_embedding = generate_embedding(asset_name, request_id=item_id)
        except Exception as exc:
            _log_stage_failure(item_id, "embed", exc)
            return _finish_with_debug(
                _error_result(echoed, _message_for_embed_exc(exc)), debug
            )
        if not name_embedding:
            return _finish_with_debug(
                _error_result(echoed, MSG_EMBED_FAILED), debug
            )
        if debug is not None:
            debug["embed_queries"].append({"stage": "name", "text": asset_name})

        if _deadline_hit(deadline):
            return _finish_with_debug(
                _error_result(echoed, MSG_TIMED_OUT), debug
            )
        try:
            candidates = query_masterdata(
                name_embedding,
                customer_id=customer_id,
                category_id=category_id,
                asset_class_id=asset_class_id,
                request_id=item_id,
            )
            if debug is not None:
                debug["pinecone_name_hits"] = _candidate_debug_rows(candidates)
                debug["best_score_name"] = _best_score(candidates)
            # A short product description can name the real master when the asset
            # name is only a size. Invoice text is left out so it cannot pull a
            # different product into the top hits.
            desc_useful = (
                bool(asset_description)
                and asset_description.casefold() != asset_name.casefold()
                and len(asset_description) <= 80
                and not _looks_like_invoice(asset_description)
                and (_best_score(candidates) or 0) < 0.75
            )
            if debug is not None:
                debug["desc_search_ran"] = False
                if not asset_description:
                    debug["desc_skip_reason"] = "no_desc"
                elif asset_description.casefold() == asset_name.casefold():
                    debug["desc_skip_reason"] = "desc_same_as_name"
                elif len(asset_description) > 80:
                    debug["desc_skip_reason"] = "desc_too_long"
                elif _looks_like_invoice(asset_description):
                    debug["desc_skip_reason"] = "invoice_heuristic"
                elif (_best_score(candidates) or 0) >= 0.75:
                    debug["desc_skip_reason"] = "strong_name_score"
                else:
                    debug["desc_skip_reason"] = None
            if desc_useful and not _deadline_hit(deadline):
                try:
                    query_text = build_embed_query(
                        asset_name,
                        asset_description,
                        max_description_length=MAX_ASSET_DESCRIPTION_LENGTH,
                        max_embed_query_length=MAX_EMBED_QUERY_LENGTH,
                    )
                    if debug is not None:
                        debug["embed_queries"].append(
                            {"stage": "name_plus_desc", "text": query_text}
                        )
                    desc_embedding = generate_embedding(
                        query_text, request_id=item_id
                    )
                    if desc_embedding:
                        described = query_masterdata(
                            desc_embedding,
                            customer_id=customer_id,
                            category_id=category_id,
                            asset_class_id=asset_class_id,
                            request_id=item_id,
                        )
                        if debug is not None:
                            debug["desc_search_ran"] = True
                            debug["desc_skip_reason"] = None
                            debug["pinecone_desc_hits"] = _candidate_debug_rows(
                                described
                            )
                        candidates = _merge_candidates(candidates, described)
                except Exception as exc:
                    _log_stage_failure(item_id, "description-search", exc)
        except Exception as exc:
            _log_stage_failure(item_id, "pinecone", exc)
            return _finish_with_debug(
                _error_result(echoed, MSG_PINECONE_UNAVAILABLE), debug
            )

    if debug is not None:
        debug["candidates_merged"] = _candidate_debug_rows(candidates)
        debug["best_score_final"] = _best_score(candidates)

    if not candidates:
        if debug is not None:
            debug["decision_path"] = "no_candidates"
        return _finish_with_debug(
            _generated_without_master(
                echoed,
                asset_name,
                asset_description,
                item_id,
                MSG_NO_CANDIDATES,
                deadline,
            ),
            debug,
        )
    best = _best_score(candidates)
    if best is not None and best < MIN_MATCH_SCORE:
        if debug is not None:
            debug["decision_path"] = "below_threshold"
        return _finish_with_debug(
            _generated_without_master(
                echoed,
                asset_name,
                asset_description,
                item_id,
                MSG_NO_ACCEPTABLE_MATCH,
                deadline,
            ),
            debug,
        )

    # Optional enhancement: LLM reviews all above-threshold candidates.
    # Default OFF keeps the existing top-score path below.
    # LLM still receives the full original AssetDescription (unchanged).
    if ENABLE_LLM_REVIEW:
        return _finish_with_debug(
            _llm_review_match(
                echoed,
                asset_name,
                asset_description,
                candidates,
                item_id,
                deadline,
                debug=debug,
            ),
            debug,
        )

    chosen = _stable_master(candidates)
    if debug is not None:
        debug["decision_path"] = "top_score"
        debug["top_score_chosen"] = _candidate_debug_rows([chosen])[0]
    return _finish_with_debug(
        _complete_match_from_candidate(
            echoed,
            chosen,
            asset_name,
            asset_description,
            item_id,
            deadline,
        ),
        debug,
    )


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


def _store_finished(
    future: Any,
    idx: int,
    work: list[tuple[int, dict[str, Any]]],
    results_slot: list[dict[str, Any] | None],
    request_id: str,
) -> None:
    if results_slot[idx] is not None or not future.done() or future.cancelled():
        return
    customer_id, asset = work[idx]
    try:
        results_slot[idx] = future.result()
    except Exception as exc:
        _log_stage_failure(f"{request_id}:worker[{idx}]", "pipeline", exc)
        results_slot[idx] = _error_result(
            _echo_input(asset, customer_id),
            MSG_PIPELINE_ERROR,
        )


def _run_work_items(
    work: list[tuple[int, dict[str, Any]]],
    *,
    request_id: str,
) -> list[dict[str, Any]]:
    """Match each asset with its own customer_id; preserve input order in flat results.

    Stops waiting at ``BATCH_DEADLINE_SEC`` and fills unfinished rows in place, so the
    host timeout is not what ends the request.
    """
    results_slot: list[dict[str, Any] | None] = [None] * len(work)
    workers = max(1, min(MAX_PARALLEL_ITEMS, len(work)))
    deadline = time.monotonic() + BATCH_DEADLINE_SEC
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(
            match_one_asset,
            asset,
            customer_id=customer_id,
            request_id=request_id,
            deadline=deadline,
        ): idx
        for idx, (customer_id, asset) in enumerate(work)
    }
    try:
        for future in as_completed(futures, timeout=max(0.0, deadline - time.monotonic())):
            _store_finished(future, futures[future], work, results_slot, request_id)
    except TimeoutError:
        logging.warning(
            "[%s] batch deadline %.0fs exceeded",
            request_id,
            BATCH_DEADLINE_SEC,
        )
    for future, idx in futures.items():
        _store_finished(future, idx, work, results_slot, request_id)
    timed_out = 0
    for future, idx in futures.items():
        if results_slot[idx] is not None:
            continue
        future.cancel()
        _store_finished(future, idx, work, results_slot, request_id)
        if results_slot[idx] is None:
            customer_id, asset = work[idx]
            results_slot[idx] = _error_result(
                _echo_input(asset, customer_id),
                MSG_TIMED_OUT,
            )
            timed_out += 1
    pool.shutdown(wait=False, cancel_futures=True)
    if timed_out:
        logging.warning("[%s] %s asset(s) marked timed out", request_id, timed_out)
    return [item for item in results_slot if item is not None]


def _http_status_for_results(results: list[dict[str, Any]]) -> int:
    """503 only when every row failed because an upstream call or the deadline did."""
    if results and all(item.get("Message") in _UPSTREAM_MESSAGES for item in results):
        return 503
    return 200


def run_asset_master_match(req: func.HttpRequest) -> func.HttpResponse:
    request_id = str(uuid.uuid4())[:8]
    logging.info("[%s] asset_master_match received", request_id)

    try:
        body = _parse_body(req)
    except ValueError as exc:
        return _json_response({"request_id": request_id, "error": str(exc)}, status=400)

    work, err = _normalize_work_items(body)
    if err:
        return _json_response({"request_id": request_id, "error": err}, status=400)

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

    # Each row echoes its own CustomerId (no mixed top-level id).
    return _json_response(
        {"request_id": request_id, "results": results},
        status=_http_status_for_results(results),
    )
