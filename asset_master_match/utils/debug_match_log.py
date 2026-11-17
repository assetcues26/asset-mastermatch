"""Disposable local match review logger.

Toggle with ENABLE_MATCH_DEBUG_LOG. Writes JSONL under Azure_function/debug/.
Delete this module + its call sites when review is done.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asset_master_match.core.config import ENABLE_MATCH_DEBUG_LOG

_LOCK = threading.Lock()
_LOG_DIR = Path(__file__).resolve().parents[2] / "debug"
_LOG_FILE = _LOG_DIR / "match_review.jsonl"


def maybe_log_match(payload: dict[str, Any]) -> None:
    """Append one review record when debug logging is enabled. Never raises."""
    if not ENABLE_MATCH_DEBUG_LOG:
        return
    try:
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with _LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        logging.exception("match debug log write failed")
