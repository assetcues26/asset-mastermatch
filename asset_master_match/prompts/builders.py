"""Build LLM prompts for master-data selection."""

from __future__ import annotations

from typing import Any

from asset_master_match.core.config import MAX_ASSET_NAME_LENGTH, MAX_PROMPT_FIELD_LENGTH
from asset_master_match.prompts.loader import load_prompt
from asset_master_match.utils.helpers import sanitize_prompt_field, to_int_or_none


def _field(value: Any, max_length: int = MAX_PROMPT_FIELD_LENGTH) -> str:
    return sanitize_prompt_field(value, max_length)


def format_candidates_for_prompt(candidates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, row in enumerate(candidates, start=1):
        lines.append(
            " | ".join(
                [
                    f"CandidateIndex={i}",
                    f"AssetClass={_field(row.get('assetclass'))}",
                    f"AssetClassId={to_int_or_none(row.get('assetclassid'))}",
                    f"Category={_field(row.get('category') or row.get('categoryname'))}",
                    f"CategoryId={to_int_or_none(row.get('categoryid'))}",
                    f"SubCategory={_field(row.get('subcategory'))}",
                    f"SubCategoryId={to_int_or_none(row.get('subcategoryid'))}",
                    f"MakeModel={_field(row.get('makemodel'))}",
                    f"MakeModelId={to_int_or_none(row.get('makemodelid'))}",
                    f"Score={row.get('score')}",
                ]
            )
        )
    return "\n".join(lines) if lines else "(none)"


def build_select_prompt(
    asset_name: str,
    candidates: list[dict[str, Any]],
    asset_description: str = "",
) -> str:
    desc = _field(asset_description, MAX_PROMPT_FIELD_LENGTH)
    return load_prompt(
        "select_master_match.md",
        asset_name=_field(asset_name, MAX_ASSET_NAME_LENGTH) or "(empty)",
        asset_description=desc or "(none)",
        candidate_summary=format_candidates_for_prompt(candidates),
    )


def build_select_candidate_prompt(
    asset_name: str,
    candidates: list[dict[str, Any]],
    asset_description: str = "",
) -> str:
    """Select-only prompt: candidate index + whether MakeModel should be generated."""
    desc = _field(asset_description, MAX_PROMPT_FIELD_LENGTH)
    return load_prompt(
        "select_candidate.md",
        asset_name=_field(asset_name, MAX_ASSET_NAME_LENGTH) or "(empty)",
        asset_description=desc or "(none)",
        candidate_summary=format_candidates_for_prompt(candidates),
    )


def build_subcategory_prompt(
    asset_name: str,
    asset_description: str = "",
) -> str:
    desc = _field(asset_description, MAX_PROMPT_FIELD_LENGTH)
    return load_prompt(
        "recommend_subcategory.md",
        asset_name=_field(asset_name, MAX_ASSET_NAME_LENGTH) or "(empty)",
        asset_description=desc or "(none)",
    )
