from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """
    Convert a value to float while preserving a caller-provided missing default.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to int while preserving a caller-provided missing default.
    """
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def clip_score(value: Any, default: float = 50.0) -> float:
    """
    Clamp a score to the 0-100 range used throughout the scoring layers.
    """
    score = safe_float(value, default)
    if score is None:
        score = default
    return round(max(0.0, min(100.0, score)), 2)


def normalize_player_name(value: Any) -> str:
    """
    Normalize player names for display and case-insensitive comparisons.
    """
    return str(value or "").strip()


def name_key(value: Any) -> str:
    """
    Return a stable lowercase key for player-name lookups.
    """
    return normalize_player_name(value).lower()
