"""Load prompt markdown templates from prompts/md/."""

from __future__ import annotations

import logging
from pathlib import Path

from asset_master_match.core.config import PROMPT_HOT_RELOAD

_MD_DIR = Path(__file__).resolve().parent / "md"
_CACHE: dict[str, str] = {}


def load_prompt(name: str, **kwargs: object) -> str:
    """Load ``name`` (with or without .md) and format with kwargs."""
    filename = name if name.endswith(".md") else f"{name}.md"
    if not PROMPT_HOT_RELOAD and filename in _CACHE:
        template = _CACHE[filename]
    else:
        path = _MD_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Prompt not found: {path}")
        template = path.read_text(encoding="utf-8")
        _CACHE[filename] = template
        logging.debug("Loaded prompt %s", filename)
    if kwargs:
        return template.format(**kwargs)
    return template
