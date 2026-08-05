from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _name_key
from draftkit.common import safe_float as _safe_float


FEATURE_SPECS = [
    {
        "feature": "projection",
        "label": "Projection",
        "columns": ["projection_component_score", "position_value_score"],
        "higher_is_better": True,
    },
    {
        "feature": "expected_finish",
        "label": "Expected Finish",
        "columns": ["expected_finish_score", "expected_finish", "projection_rank"],
        "higher_is_better": False,
        "rank_like": True,
    },
    {
        "feature": "workload",
        "label": "Workload",
        "columns": ["workload_score", "workload", "touch_projection", "snap_share"],
        "higher_is_better": True,
    },
    {
        "feature": "durability",
        "label": "Durability",
        "columns": ["durability_grade", "durability", "durability_score"],
        "higher_is_better": True,
    },
    {
        "feature": "offensive_environment",
        "label": "Offensive Environment",
        "columns": [
            "offensive_environment_score",
            "team_offense_score",
            "offense_score",
            "team_context_score",
        ],
        "higher_is_better": True,
    },
    {
        "feature": "adp_value",
        "label": "ADP Value",
        "columns": ["adp_value_component_score", "value_score"],
        "higher_is_better": True,
    },
    {
        "feature": "championship_equity",
        "label": "Championship Equity",
        "columns": [
            "championship_equity_score",
            "championship_equity",
            "league_winning_upside",
        ],
        "higher_is_better": True,
    },
    {
        "feature": "future_availability",
        "label": "Future Availability",
        "columns": [
            "drafted_before_next_pick_probability",
            "availability_score",
            "survival_probability",
        ],
        "higher_is_better": True,
    },
    {
        "feature": "risk",
        "label": "Risk",
        "columns": ["risk_score", "injury_risk"],
        "higher_is_better": False,
    },
]


def _merge_optional(
    base_df: pd.DataFrame,
    optional_df: Optional[pd.DataFrame],
    columns: Iterable[str],
) -> pd.DataFrame:
    if optional_df is None or optional_df.empty or "player_name" not in optional_df.columns:
        return base_df

    subset_cols = ["player_name"] + [
        column for column in columns if column in optional_df.columns
    ]
    if len(subset_cols) <= 1:
        return base_df

    out = base_df.copy()
    out["_player_key"] = out["player_name"].apply(_name_key)
    subset = optional_df[subset_cols].copy()
    subset["_player_key"] = subset["player_name"].apply(_name_key)
    subset = subset.drop(columns=["player_name"])
    out = out.merge(subset, on="_player_key", how="left", suffixes=("", "_optional"))
    return out.drop(columns=["_player_key"])


def _rank_to_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([pd.NA] * len(series), index=series.index)

    min_rank = float(valid.min())
    max_rank = float(valid.max())
    if max_rank <= min_rank:
        return pd.Series([100.0 if pd.notna(value) else pd.NA for value in numeric], index=series.index)

    return numeric.apply(
        lambda value: pd.NA
        if pd.isna(value)
        else round(100.0 - ((float(value) - min_rank) / (max_rank - min_rank) * 100.0), 2)
    )


def _column_score(df: pd.DataFrame, spec: Dict[str, Any]) -> Tuple[pd.Series, Optional[str]]:
    for column in spec["columns"]:
        if column not in df.columns:
            continue

        if spec.get("rank_like") and column in {"projection_rank", "expected_finish"}:
            scored = _rank_to_score(df[column])
        else:
            scored = pd.to_numeric(df[column], errors="coerce")
            if not spec.get("higher_is_better", True):
                scored = scored.apply(
                    lambda value: pd.NA
                    if pd.isna(value)
                    else round(100.0 - _clip_score(value), 2)
                )

        if scored.notna().any():
            return scored.apply(lambda value: pd.NA if pd.isna(value) else _clip_score(value)), column

    return pd.Series([pd.NA] * len(df), index=df.index), None


def _corr(xs: pd.Series, ys: pd.Series) -> float:
    paired = pd.DataFrame({"x": xs, "y": ys}).dropna()
    if len(paired) < 2:
        return 0.0
    if paired["x"].nunique() <= 1 or paired["y"].nunique() <= 1:
        return 0.0
    return float(paired["x"].corr(paired["y"]))


def _variance(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    return float(numeric.var())


def build_top_100_component_audit(
    recommendations_df: pd.DataFrame,
    championship_equity_df: Optional[pd.DataFrame] = None,
    availability_df: Optional[pd.DataFrame] = None,
    top_n: int = 100,
) -> Dict[str, Any]:
    if recommendations_df is None or recommendations_df.empty:
        return {
            "player_contributions": pd.DataFrame(),
            "feature_influence": pd.DataFrame(),
            "variance_flags": [],
            "unsupported_features": [spec["label"] for spec in FEATURE_SPECS],
        }

    working_df = recommendations_df.head(top_n).copy().reset_index(drop=True)
    if "player_name" not in working_df.columns:
        return {
            "player_contributions": pd.DataFrame(),
            "feature_influence": pd.DataFrame(),
            "variance_flags": [],
            "unsupported_features": [spec["label"] for spec in FEATURE_SPECS],
        }

    working_df = _merge_optional(
        working_df,
        championship_equity_df,
        ["championship_equity_score", "championship_equity", "league_winning_upside"],
    )
    working_df = _merge_optional(
        working_df,
        availability_df,
        [
            "drafted_before_next_pick_probability",
            "availability_probability",
            "survival_probability",
        ],
    )

    target = pd.to_numeric(working_df.get("final_score"), errors="coerce")
    feature_rows: List[Dict[str, Any]] = []
    feature_score_cols = []

    for spec in FEATURE_SPECS:
        score_series, source_column = _column_score(working_df, spec)
        score_col = f"{spec['feature']}_score"
        working_df[score_col] = score_series
        feature_score_cols.append(score_col)

        corr = _corr(score_series, target)
        variance = _variance(score_series)
        raw_influence = abs(corr) * (variance ** 0.5 if variance > 0 else 0.0)
        feature_rows.append({
            "feature": spec["feature"],
            "feature_label": spec["label"],
            "source_column": source_column or "unsupported",
            "supported": source_column is not None,
            "average_feature_score": round(_safe_float(score_series.mean(), 0.0), 2),
            "average_absolute_influence": round(raw_influence, 4),
            "correlation_to_final_score": round(corr, 4),
            "feature_variance": round(variance, 4),
            "raw_influence": raw_influence,
        })

    total_influence = sum(row["raw_influence"] for row in feature_rows)
    total_r2 = sum(row["correlation_to_final_score"] ** 2 for row in feature_rows)

    for row in feature_rows:
        row["average_influence_pct"] = (
            round((row["raw_influence"] / total_influence) * 100.0, 2)
            if total_influence > 0
            else 0.0
        )
        row["score_variance_share_pct"] = (
            round(((row["correlation_to_final_score"] ** 2) / total_r2) * 100.0, 2)
            if total_r2 > 0
            else 0.0
        )
        row["contributes_over_20_pct_variance"] = row["score_variance_share_pct"] > 20.0
        row.pop("raw_influence", None)

    influence_df = pd.DataFrame(feature_rows).sort_values(
        ["average_influence_pct", "score_variance_share_pct"],
        ascending=False,
    ).reset_index(drop=True)

    contribution_weights = {
        row["feature"]: row["average_influence_pct"] / 100.0
        for row in feature_rows
    }

    player_rows = []
    for rank, row in working_df.iterrows():
        out = {
            "rank": rank + 1,
            "player_name": row.get("player_name"),
            "position": row.get("position"),
            "team": row.get("team"),
            "final_recommendation_score": round(_safe_float(row.get("final_score"), 0.0), 2),
        }

        weighted_parts = {}
        for spec in FEATURE_SPECS:
            feature = spec["feature"]
            score_value = _safe_float(row.get(f"{feature}_score"), None)
            score_output = None if score_value is None else round(score_value, 2)
            contribution = 0.0 if score_value is None else score_value * contribution_weights.get(feature, 0.0)
            weighted_parts[feature] = contribution
            out[f"{feature}_score"] = score_output

        contribution_total = sum(abs(value) for value in weighted_parts.values())
        for feature, value in weighted_parts.items():
            out[f"{feature}_contribution_pct"] = (
                round((abs(value) / contribution_total) * 100.0, 2)
                if contribution_total > 0
                else 0.0
            )

        player_rows.append(out)

    player_df = pd.DataFrame(player_rows)
    variance_flags = influence_df[
        influence_df["contributes_over_20_pct_variance"] == True
    ][["feature_label", "score_variance_share_pct", "source_column"]].to_dict("records")
    unsupported = influence_df[
        influence_df["supported"] == False
    ]["feature_label"].tolist()

    return {
        "player_contributions": player_df,
        "feature_influence": influence_df,
        "variance_flags": variance_flags,
        "unsupported_features": unsupported,
    }
