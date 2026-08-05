from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    roc_auc_score = None


VALIDATION_DIR = Path(__file__).resolve().parent
INPUT_PRIMARY = VALIDATION_DIR / "predraft_validation_dataset_age_edge_score.csv"
INPUT_FALLBACK = VALIDATION_DIR / "predraft_validation_dataset_projected.csv"
OUTPUT = VALIDATION_DIR / "predraft_validation_dataset_late_role_growth_score.csv"
FEATURE_REPORT = VALIDATION_DIR / "late_season_role_growth_score_v1_report.md"
VALIDATION_OUTPUT = VALIDATION_DIR / "late_season_role_growth_score_v1_validation.csv"
VALIDATION_REPORT = VALIDATION_DIR / "late_season_role_growth_score_v1_validation_report.md"

PREFERRED_TARGETS = [
    "WR_Beat_ADP_By_12",
    "RB_Beat_ADP_By_12",
    "WR_Underpriced_Top24",
    "RB_Underpriced_Top24",
    "WR_Underpriced_Top12",
    "RB_Underpriced_Top12",
]

ADP_WINDOWS = [
    ("1-24", 1, 24),
    ("25-48", 25, 48),
    ("49-72", 49, 72),
    ("73-96", 73, 96),
    ("97-120", 97, 120),
    ("121-150", 121, 150),
    ("151+", 151, math.inf),
]

POSITION_WEIGHTS = {
    "WR": {
        "target_growth_component": 0.35,
        "route_growth_component": 0.25,
        "snap_growth_component": 0.15,
        "red_zone_growth_component": 0.15,
        "direct_late_role_growth_component": 0.10,
    },
    "RB": {
        "touch_growth_component": 0.35,
        "target_growth_component": 0.20,
        "snap_growth_component": 0.20,
        "red_zone_growth_component": 0.15,
        "direct_late_role_growth_component": 0.10,
    },
    "TE": {
        "route_growth_component": 0.30,
        "target_growth_component": 0.30,
        "snap_growth_component": 0.15,
        "red_zone_growth_component": 0.15,
        "direct_late_role_growth_component": 0.10,
    },
    "QB": {
        "direct_late_role_growth_component": 1.00,
    },
}

UNSAFE_TARGET_PATTERNS = re.compile(r"beat_adp|underpriced|breakout|tier_jump|top12|top24|final_|finish|fantasy_points$", re.I)


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def normalize_position(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"WR", "RB", "TE", "QB"} else text


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def detect_targets(df: pd.DataFrame) -> list[str]:
    found = [target for target in PREFERRED_TARGETS if target in df.columns]
    feature_prefixes = ("age_curve_", "late_season_role_growth", "late_role_")
    for col in df.columns:
        lower = col.lower()
        if col in found or lower.startswith(feature_prefixes):
            continue
        if re.search(r"Beat_ADP|Underpriced|Breakout|Tier_Jump", col, re.I):
            found.append(col)
    return found


def infer_target_position(target: str) -> str | None:
    upper = target.upper()
    for pos in ("WR", "RB", "TE", "QB"):
        if upper.startswith(pos + "_"):
            return pos
    return None


def detect_columns(df: pd.DataFrame, includes: list[str], excludes: list[str] | None = None) -> list[str]:
    excludes = excludes or []
    out = []
    for col in df.columns:
        lower = col.lower()
        if UNSAFE_TARGET_PATTERNS.search(lower):
            continue
        if all(term in lower for term in includes) and not any(term in lower for term in excludes):
            out.append(col)
    return out


def likely_source_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    direct = []
    for col in df.columns:
        lower = col.lower()
        if UNSAFE_TARGET_PATTERNS.search(lower):
            continue
        if ("late" in lower or "second_half" in lower) and "growth" in lower:
            direct.append(col)
    return {
        "target_growth_component": sorted(set(
            detect_columns(df, ["target", "growth"])
            + detect_columns(df, ["targets", "growth"])
            + detect_columns(df, ["second_half", "target"])
            + detect_columns(df, ["late", "target"])
        )),
        "touch_growth_component": sorted(set(
            detect_columns(df, ["touch", "growth"])
            + detect_columns(df, ["carry", "growth"])
            + detect_columns(df, ["carries", "growth"])
            + detect_columns(df, ["second_half", "touch"])
            + detect_columns(df, ["late", "touch"])
            + detect_columns(df, ["late", "carr"])
        )),
        "snap_growth_component": sorted(set(
            detect_columns(df, ["snap", "growth"])
            + detect_columns(df, ["second_half", "snap"])
            + detect_columns(df, ["late", "snap"])
        )),
        "route_growth_component": sorted(set(
            detect_columns(df, ["route", "growth"])
            + detect_columns(df, ["second_half", "route"])
            + detect_columns(df, ["late", "route"])
        )),
        "red_zone_growth_component": sorted(set(
            detect_columns(df, ["red_zone", "growth"])
            + detect_columns(df, ["red", "zone", "growth"])
            + detect_columns(df, ["late", "red_zone"])
            + detect_columns(df, ["late", "red", "zone"])
        )),
        "direct_late_role_growth_component": sorted(set(direct + detect_columns(df, ["late_season_role_growth"]))),
    }


def percentile_score_within_group(df: pd.DataFrame, value: pd.Series, season_col: str, position_col: str) -> pd.Series:
    numeric = safe_numeric(value)
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=df.index)
    temp = pd.DataFrame({"season": df[season_col], "position": df[position_col], "value": numeric})
    return temp.groupby(["season", "position"])["value"].rank(pct=True) * 100.0


def build_growth_component(df: pd.DataFrame, cols: list[str], season_col: str, position_col: str) -> pd.Series:
    usable = [col for col in cols if col in df.columns and safe_numeric(df[col]).notna().any()]
    if not usable:
        return pd.Series(np.nan, index=df.index)
    component_parts = []
    for col in usable:
        component_parts.append(percentile_score_within_group(df, safe_numeric(df[col]), season_col, position_col))
    return pd.concat(component_parts, axis=1).mean(axis=1, skipna=True)


def combine_growth_components(df: pd.DataFrame, position_col: str) -> pd.Series:
    scores = []
    for idx, row in df.iterrows():
        pos = normalize_position(row.get(position_col))
        weights = POSITION_WEIGHTS.get(pos, POSITION_WEIGHTS["WR"])
        weighted = 0.0
        weight_sum = 0.0
        for component, weight in weights.items():
            val = row.get(component)
            if pd.notna(val):
                weighted += float(val) * weight
                weight_sum += weight
        scores.append(weighted / weight_sum if weight_sum > 0 else np.nan)
    return pd.Series(scores, index=df.index).clip(0, 100)


def assign_late_role_growth_bucket(score: Any) -> str:
    if pd.isna(score):
        return "Unknown"
    score = float(score)
    if score >= 85:
        return "Major Late-Season Role Growth"
    if score >= 70:
        return "Strong Late-Season Role Growth"
    if score >= 55:
        return "Moderate Late-Season Role Growth"
    if score >= 40:
        return "Stable Role"
    return "Role Decline"


def build_late_role_growth_note(row: pd.Series) -> str:
    bucket = row["late_season_role_growth_bucket"]
    count = int(row["late_season_role_growth_component_count"]) if pd.notna(row["late_season_role_growth_component_count"]) else 0
    if bucket == "Unknown":
        return "No usable pre-draft-safe late-season role growth columns were available."
    return f"{bucket} based on {count} available pre-draft-safe growth component(s)."


def top_decile_hit_rate(frame: pd.DataFrame, score_col: str, target: str) -> float:
    if frame.empty or score_col not in frame.columns:
        return np.nan
    scored = frame[[score_col, target]].copy()
    scored[score_col] = safe_numeric(scored[score_col])
    scored[target] = safe_numeric(scored[target])
    scored = scored.dropna()
    if scored.empty:
        return np.nan
    top_n = max(1, int(math.ceil(len(scored) * 0.10)))
    return float(scored.sort_values(score_col, ascending=False).head(top_n)[target].mean())


def auc_or_nan(y_true: pd.Series, scores: pd.Series) -> float:
    if roc_auc_score is None:
        return np.nan
    y = safe_numeric(y_true)
    s = safe_numeric(scores)
    mask = y.notna() & s.notna()
    if mask.sum() < 2 or y[mask].nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(y[mask], s[mask]))
    except Exception:
        return np.nan


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> tuple[pd.Series, str]:
    missing = [f for f in features if f not in train.columns or not train[f].notna().any() or not test[f].notna().any()]
    if missing:
        return pd.Series(np.nan, index=test.index), "skipped_required_feature_missing:" + ",".join(missing)
    available = list(features)
    if not available:
        return pd.Series(np.nan, index=test.index), "skipped_no_features"
    y = safe_numeric(train[target])
    if y.nunique(dropna=True) < 2:
        return pd.Series(np.nan, index=test.index), "skipped_one_class_train"
    train_x = train[available].apply(pd.to_numeric, errors="coerce")
    test_x = test[available].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(numeric_only=True).fillna(0.0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    if LogisticRegression is None or Pipeline is None or StandardScaler is None:
        ranked = test_x.rank(pct=True).mean(axis=1)
        return ranked, "fallback_rank_score_no_sklearn"
    try:
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ])
        estimator.fit(train_x, y)
        return pd.Series(estimator.predict_proba(test_x)[:, 1], index=test.index), "fit"
    except Exception as exc:
        ranked = test_x.rank(pct=True).mean(axis=1)
        return ranked, f"fallback_rank_score_error:{exc.__class__.__name__}"


def model_feature_groups(adp_col: str | None, age_col: str | None) -> dict[str, list[str]]:
    groups = {
        "ADP-only": [adp_col] if adp_col else [],
        "late-season role growth only": ["late_season_role_growth_score"],
        "ADP + late-season role growth": ([adp_col] if adp_col else []) + ["late_season_role_growth_score"],
    }
    if age_col:
        groups["ADP + age"] = ([adp_col] if adp_col else []) + [age_col]
        groups["ADP + age + late-season role growth"] = ([adp_col] if adp_col else []) + [age_col, "late_season_role_growth_score"]
    return {k: [f for f in v if f] for k, v in groups.items() if v}


def summarize_predictions(predictions: pd.DataFrame, target: str, model_name: str, position: str | None, scope: str, window: str | None = None) -> dict[str, Any]:
    model_rows = predictions[predictions["model_name"] == model_name].copy()
    if model_rows.empty or model_rows["prediction"].notna().sum() == 0:
        return {
            "result_type": scope,
            "draft_window": window or "",
            "target": target,
            "position": position or "ALL",
            "model_name": model_name,
            "rows": int(len(model_rows)),
            "status": "skipped_no_valid_predictions",
        }
    adp_rows = predictions[predictions["model_name"] == "ADP-only"].copy()
    yearly = []
    for season, season_rows in model_rows.groupby("test_season"):
        base_rows = adp_rows[adp_rows["test_season"] == season]
        model_hit = top_decile_hit_rate(season_rows, "prediction", target)
        adp_hit = top_decile_hit_rate(base_rows, "prediction", target)
        yearly.append({
            "season": season,
            "model_hit": model_hit,
            "adp_hit": adp_hit,
            "lift": model_hit - adp_hit if pd.notna(model_hit) and pd.notna(adp_hit) else np.nan,
        })
    yearly_df = pd.DataFrame(yearly)
    valid_lifts = yearly_df["lift"].dropna() if not yearly_df.empty else pd.Series(dtype=float)
    rows = len(model_rows)
    return {
        "result_type": scope,
        "draft_window": window or "",
        "target": target,
        "position": position or "ALL",
        "model_name": model_name,
        "rows": int(rows),
        "positive_rate": float(safe_numeric(model_rows[target]).mean()) if rows else np.nan,
        "seasons_tested": int(model_rows["test_season"].nunique()) if rows else 0,
        "auc": auc_or_nan(model_rows[target], model_rows["prediction"]),
        "top_decile_hit_rate": top_decile_hit_rate(model_rows, "prediction", target),
        "baseline_hit_rate": top_decile_hit_rate(adp_rows, "prediction", target) if not adp_rows.empty else np.nan,
        "lift_over_baseline": (
            top_decile_hit_rate(model_rows, "prediction", target) - top_decile_hit_rate(adp_rows, "prediction", target)
            if not adp_rows.empty else np.nan
        ),
        "improvement_over_adp_only": (
            top_decile_hit_rate(model_rows, "prediction", target) - top_decile_hit_rate(adp_rows, "prediction", target)
            if model_name != "ADP-only" and not adp_rows.empty else 0.0
        ),
        "seasons_where_model_beat_adp_only": int((valid_lifts > 0).sum()) if model_name != "ADP-only" else 0,
        "average_yearly_lift": float(valid_lifts.mean()) if len(valid_lifts) else np.nan,
        "median_yearly_lift": float(valid_lifts.median()) if len(valid_lifts) else np.nan,
        "status": "evaluated",
    }


def walk_forward_predictions(df: pd.DataFrame, target: str, feature_groups: dict[str, list[str]], season_col: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    skipped = []
    seasons = sorted(int(s) for s in safe_numeric(df[season_col]).dropna().unique())
    for test_season in seasons:
        train = df[safe_numeric(df[season_col]) < test_season].copy()
        test = df[safe_numeric(df[season_col]) == test_season].copy()
        train_y = safe_numeric(train[target])
        test_y = safe_numeric(test[target])
        if len(train) < 50 or train_y.sum() < 5 or (len(train_y.dropna()) - train_y.sum()) < 5:
            skipped.append({"target": target, "test_season": test_season, "reason": "insufficient_training_rows_or_classes"})
            continue
        if len(test) < 20 or test_y.sum() < 2:
            skipped.append({"target": target, "test_season": test_season, "reason": "insufficient_test_rows_or_positives"})
            continue
        for model_name, features in feature_groups.items():
            pred, status = fit_predict(train, test, target, features)
            part = test.copy()
            part["prediction"] = pred
            part["model_name"] = model_name
            part["test_season"] = test_season
            part["fit_status"] = status
            rows.append(part)
    if not rows:
        return pd.DataFrame(), skipped
    return pd.concat(rows, ignore_index=True), skipped


def run_walk_forward_validation(df: pd.DataFrame, targets: list[str], season_col: str, position_col: str, adp_col: str | None, age_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    predictions_all = []
    for target in targets:
        target_pos = infer_target_position(target)
        base = df.copy()
        if target_pos:
            base = base[base[position_col].map(normalize_position) == target_pos].copy()
        feature_groups = model_feature_groups(adp_col, age_col)
        needed = sorted(set(sum(feature_groups.values(), []) + [target, season_col]))
        select_cols = []
        for candidate in needed + [position_col, "player_name", "overall_adp"]:
            if candidate in base.columns and candidate not in select_cols:
                select_cols.append(candidate)
        base = base[select_cols].copy()
        for col in needed:
            if col in base.columns:
                base[col] = safe_numeric(base[col])
        base = base[base[target].notna()].copy()
        if len(base) < 100:
            results.append({"result_type": "full_pool", "target": target, "position": target_pos or "ALL", "model_name": "", "status": "skipped_insufficient_full_pool_rows", "rows": len(base)})
            continue
        preds, skipped = walk_forward_predictions(base, target, feature_groups, season_col)
        for item in skipped:
            item.update({"result_type": "season_skip", "position": target_pos or "ALL", "model_name": "", "rows": np.nan})
            results.append(item)
        if preds.empty:
            results.append({"result_type": "full_pool", "target": target, "position": target_pos or "ALL", "model_name": "", "status": "skipped_no_walk_forward_predictions", "rows": len(base)})
            continue
        predictions_all.append(preds)
        for model_name in feature_groups:
            results.append(summarize_predictions(preds, target, model_name, target_pos, "full_pool"))
    pred_out = pd.concat(predictions_all, ignore_index=True) if predictions_all else pd.DataFrame()
    return pd.DataFrame(results), pred_out


def run_bucket_validation(df: pd.DataFrame, targets: list[str], position_col: str) -> pd.DataFrame:
    rows = []
    for target in targets:
        target_pos = infer_target_position(target)
        base = df[df[position_col].map(normalize_position) == target_pos].copy() if target_pos else df.copy()
        base = base[base[target].notna()].copy()
        baseline = float(safe_numeric(base[target]).mean()) if len(base) else np.nan
        for bucket, group in base.groupby("late_season_role_growth_bucket", dropna=False):
            y = safe_numeric(group[target])
            rows.append({
                "result_type": "bucket",
                "target": target,
                "position": target_pos or "ALL",
                "bucket": bucket,
                "rows": int(len(group)),
                "positives": int(y.sum()) if y.notna().any() else 0,
                "positive_rate": float(y.mean()) if y.notna().any() else np.nan,
                "baseline_positive_rate": baseline,
                "lift_over_baseline": float(y.mean() - baseline) if y.notna().any() and pd.notna(baseline) else np.nan,
            })
        valid = base[base["late_season_role_growth_score"].notna()].copy()
        if len(valid) >= 100:
            valid["score_decile"] = pd.qcut(valid["late_season_role_growth_score"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
            for decile, group in valid.groupby("score_decile", dropna=False):
                y = safe_numeric(group[target])
                rows.append({
                    "result_type": "score_decile",
                    "target": target,
                    "position": target_pos or "ALL",
                    "bucket": f"decile_{int(decile)}" if pd.notna(decile) else "Unknown",
                    "rows": int(len(group)),
                    "positives": int(y.sum()) if y.notna().any() else 0,
                    "positive_rate": float(y.mean()) if y.notna().any() else np.nan,
                    "baseline_positive_rate": baseline,
                    "lift_over_baseline": float(y.mean() - baseline) if y.notna().any() and pd.notna(baseline) else np.nan,
                })
    return pd.DataFrame(rows)


def run_draft_window_validation(df: pd.DataFrame, targets: list[str], season_col: str, position_col: str, adp_col: str | None, age_col: str | None) -> pd.DataFrame:
    if not adp_col:
        return pd.DataFrame([{"result_type": "draft_window", "status": "skipped_no_adp_column"}])
    rows = []
    adp = safe_numeric(df[adp_col])
    for label, low, high in ADP_WINDOWS:
        if math.isinf(high):
            subset = df[adp >= low].copy()
        else:
            subset = df[(adp >= low) & (adp <= high)].copy()
        if len(subset) < 40:
            rows.append({"result_type": "draft_window", "draft_window": label, "status": "skipped_insufficient_window_rows", "rows": len(subset)})
            continue
        full_results, _ = run_walk_forward_validation(subset, targets, season_col, position_col, adp_col, age_col)
        if full_results.empty:
            rows.append({"result_type": "draft_window", "draft_window": label, "status": "skipped_no_results", "rows": len(subset)})
            continue
        full_results["result_type"] = "draft_window"
        full_results["draft_window"] = label
        rows.extend(full_results.to_dict("records"))
    return pd.DataFrame(rows)


def classify_signal_strength(validation: pd.DataFrame) -> tuple[str, str]:
    evaluated = validation[(validation["result_type"] == "full_pool") & (validation["status"] == "evaluated")].copy()
    if evaluated.empty:
        return "No Signal", "Validation could not produce enough walk-forward results."
    late = evaluated[evaluated["model_name"] == "ADP + late-season role growth"]
    combo = evaluated[evaluated["model_name"] == "ADP + age + late-season role growth"]
    candidate = pd.concat([late, combo], ignore_index=True)
    candidate["improvement_over_adp_only"] = safe_numeric(candidate.get("improvement_over_adp_only", pd.Series(dtype=float)))
    candidate["seasons_where_model_beat_adp_only"] = safe_numeric(candidate.get("seasons_where_model_beat_adp_only", pd.Series(dtype=float)))
    positive = candidate[(candidate["improvement_over_adp_only"] > 0) & (candidate["seasons_where_model_beat_adp_only"] >= 2)]
    windows = validation[(validation["result_type"] == "draft_window") & (validation["status"] == "evaluated")].copy()
    windows["improvement_over_adp_only"] = safe_numeric(windows.get("improvement_over_adp_only", pd.Series(dtype=float)))
    useful_windows = windows[
        windows["model_name"].isin(["ADP + late-season role growth", "ADP + age + late-season role growth"])
        & (windows["improvement_over_adp_only"] > 0)
        & (safe_numeric(windows.get("seasons_where_model_beat_adp_only", pd.Series(dtype=float))) >= 2)
    ]
    if positive.empty:
        return "No Signal", "ADP plus late-season role growth did not repeatably improve over ADP-only."
    if not useful_windows.empty and positive["improvement_over_adp_only"].max() >= 0.10:
        return "Tie-Breaker Only", "There is modest positive full-pool and draft-window evidence, but it is not app-ready."
    return "Weak Research Signal", "Some full-pool improvement exists, but draft-window repeatability is weak or limited."



def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "No rows."
    out = frame.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    out = out.fillna("")
    cols = list(out.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in out.iterrows():
        values = [str(row[c]).replace("|", "/") for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def write_feature_report(
    df: pd.DataFrame,
    input_path: Path,
    diagnostics: dict[str, Any],
    component_cols: list[str],
) -> None:
    lines = [
        "# Late-Season Role Growth Score V1",
        "",
        "## Executive summary",
        "Late-Season Role Growth Score V1 was built as a research-only pre-draft-safe feature. It preserves existing age and projection fields and does not modify any app-facing files.",
        "",
        "The available dataset has limited explicit late-season usage data. The only strong discovered late-season source column is primarily WR-specific, so RB coverage is expected to be sparse or unavailable.",
        "",
        "## Input/output files",
        f"- Input: `{input_path}`",
        f"- Output: `{OUTPUT}`",
        f"- Validation output: `{VALIDATION_OUTPUT}`",
        "",
        "## Columns detected",
        f"- Season column: `{diagnostics['season_col']}`",
        f"- Player column: `{diagnostics['player_col']}`",
        f"- Position column: `{diagnostics['position_col']}`",
        f"- ADP column: `{diagnostics['adp_col']}`",
        f"- Age score column: `{diagnostics['age_col']}`",
        "",
        "## Source columns used",
    ]
    for component, cols in diagnostics["source_columns_used"].items():
        lines.append(f"- {component}: {', '.join(f'`{c}`' for c in cols) if cols else 'none'}")
    lines.extend([
        "",
        "## Source columns missing/skipped",
    ])
    for component, cols in diagnostics["source_columns_skipped"].items():
        lines.append(f"- {component}: {', '.join(cols) if cols else 'none'}")
    lines.extend([
        "",
        "## Scoring logic by position",
        "- WR emphasizes target growth, route growth, snap growth, red-zone target growth, and direct late-role growth.",
        "- RB emphasizes touch growth, target growth, snap growth, red-zone touch growth, and direct late-role growth.",
        "- TE uses target, route, snap, red-zone, and direct growth if rows exist.",
        "- QB is not forced unless direct late-role growth data exists.",
        "- Component weights are renormalized when components are missing. Missing source data becomes NaN, not zero.",
        "",
        "## Component weights",
    ])
    for pos, weights in POSITION_WEIGHTS.items():
        rendered = ", ".join(f"{k}: {v:.0%}" for k, v in weights.items())
        lines.append(f"- {pos}: {rendered}")
    lines.extend([
        "",
        "## Missingness diagnostics",
    ])
    for col in component_cols + ["late_season_role_growth_score"]:
        miss = df[col].isna().mean() if col in df.columns else 1.0
        lines.append(f"- {col}: {miss:.1%} missing")
    lines.extend([
        "",
        "## Score distribution by position",
    ])
    dist = diagnostics["score_distribution_by_position"]
    lines.append(markdown_table(dist) if not dist.empty else "No valid scores.")
    lines.extend([
        "",
        "## Bucket definitions",
        "- Major Late-Season Role Growth: score >= 85",
        "- Strong Late-Season Role Growth: 70 to 84.99",
        "- Moderate Late-Season Role Growth: 55 to 69.99",
        "- Stable Role: 40 to 54.99",
        "- Role Decline: below 40",
        "- Unknown: not enough source data",
        "",
        "## Data concerns",
        "- Coverage is narrow because most requested late-season opportunity columns are not present in the current dataset.",
        "- `prior_wr_late_season_target_growth` is pre-draft-safe because it describes the prior season, but it is position-specific and should not be generalized to RB without RB-specific usage data.",
        "- No Beat_ADP, Underpriced, Breakout, Tier_Jump, final ranking, or final fantasy outcome columns are used in score construction.",
        "",
        "## Research-only status",
        "This feature is not app-ready and was not promoted into production.",
    ])
    FEATURE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(validation: pd.DataFrame, classification: str, reason: str, targets: list[str], concerns: list[str]) -> None:
    evaluated = validation[(validation.get("status") == "evaluated") if "status" in validation.columns else []].copy()
    best_full = evaluated[evaluated["result_type"] == "full_pool"].sort_values("improvement_over_adp_only", ascending=False).head(8) if not evaluated.empty else pd.DataFrame()
    best_window = evaluated[evaluated["result_type"] == "draft_window"].sort_values("improvement_over_adp_only", ascending=False).head(8) if not evaluated.empty else pd.DataFrame()
    late = evaluated[evaluated["model_name"] == "ADP + late-season role growth"] if not evaluated.empty else pd.DataFrame()
    combo = evaluated[evaluated["model_name"] == "ADP + age + late-season role growth"] if not evaluated.empty else pd.DataFrame()
    lines = [
        "# Late-Season Role Growth Score V1 Validation",
        "",
        "## Executive summary",
        f"Final classification: **{classification}**.",
        "",
        reason,
        "",
        "This is research-only. The score was not promoted to the app, Draft Mode, rankings, recommendations, or any production-facing file.",
        "",
        "## Targets tested",
        "\n".join(f"- `{target}`" for target in targets) if targets else "No targets were available.",
        "",
        "## Validation methodology",
        "Validation used season-based walk-forward splits only. Each test season was predicted using earlier seasons. Models were simple LogisticRegression pipelines with median imputation and standardization.",
        "",
        "Minimum sample rules were applied for full-pool, per-season, and draft-window tests. Skipped rows are retained in the validation CSV with a skip reason where applicable.",
        "",
        "## Bucket-level validation",
        "Bucket and score-decile target rates are included in the validation CSV as `bucket` and `score_decile` result types.",
        "",
        "## ADP-only vs ADP + late-season role growth",
    ]
    if late.empty:
        lines.append("No evaluated ADP + late-season role growth result cleared the sample rules.")
    else:
        lines.append(markdown_table(late.sort_values("improvement_over_adp_only", ascending=False).head(10)))
    lines.extend([
        "",
        "## ADP + age vs ADP + age + late-season role growth",
    ])
    if combo.empty:
        lines.append("No evaluated ADP + age + late-season role growth result cleared the sample rules, or age score was unavailable.")
    else:
        lines.append(markdown_table(combo.sort_values("improvement_over_adp_only", ascending=False).head(10)))
    lines.extend([
        "",
        "## Draft-window validation",
    ])
    lines.append(markdown_table(best_window) if not best_window.empty else "No draft-window result cleared the repeatability threshold.")
    lines.extend([
        "",
        "## Season repeatability",
        "Repeatability is measured by `seasons_where_model_beat_adp_only`, `average_yearly_lift`, and `median_yearly_lift` in the validation CSV.",
        "",
        "## Best result",
    ])
    lines.append(markdown_table(best_full) if not best_full.empty else "No evaluated full-pool result.")
    lines.extend([
        "",
        "## Independent signal beyond ADP",
        "The score only has independent signal if ADP + late-season role growth beats ADP-only repeatedly across seasons and survives draft-window checks. Do not infer edge from a single positive average.",
        "",
        "## Data concerns",
    ])
    lines.extend(f"- {concern}" for concern in concerns)
    lines.extend([
        "",
        "## Final classification",
        f"**{classification}**: {reason}",
        "",
        "## Recommended next research step",
        "Add real prior-season weekly opportunity data for RB and WR, especially weeks 10-17 targets, routes, snaps, carries, red-zone touches, and team-level opportunity changes. Then rebuild this score with balanced position coverage and rerun the same ADP walk-forward validation.",
    ])
    VALIDATION_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_path = INPUT_PRIMARY if INPUT_PRIMARY.exists() else INPUT_FALLBACK
    if not input_path.exists():
        raise FileNotFoundError(f"No input dataset found at {INPUT_PRIMARY} or {INPUT_FALLBACK}")

    df = pd.read_csv(input_path)
    season_col = detect_column(df, ["season"])
    player_col = detect_column(df, ["player_name", "player", "name"])
    position_col = detect_column(df, ["position", "pos"])
    adp_col = detect_column(df, ["overall_adp", "preseason_adp", "adp"])
    positional_adp_col = detect_column(df, ["positional_adp"])
    age_col = detect_column(df, ["age_curve_edge_score", "age_curve_score"])

    if not season_col or not position_col:
        raise ValueError("Dataset must contain season and position columns.")
    if position_col != "position":
        df["position"] = df[position_col]
        position_col = "position"

    df[position_col] = df[position_col].map(normalize_position)
    source_cols = likely_source_columns(df)
    used = {component: [col for col in cols if safe_numeric(df[col]).notna().any()] for component, cols in source_cols.items()}
    skipped = {component: [col for col in cols if col not in used[component]] for component, cols in source_cols.items()}

    component_cols = sorted({component for weights in POSITION_WEIGHTS.values() for component in weights})
    for component in component_cols:
        df[component] = build_growth_component(df, used.get(component, []), season_col, position_col)

    df["late_season_role_growth_component_count"] = df[component_cols].notna().sum(axis=1)
    df["late_season_role_growth_score"] = combine_growth_components(df, position_col)
    df.loc[df["late_season_role_growth_component_count"] == 0, "late_season_role_growth_score"] = np.nan
    df["late_season_role_growth_bucket"] = df["late_season_role_growth_score"].map(assign_late_role_growth_bucket)
    df["late_season_role_growth_source_confidence"] = np.select(
        [
            df["late_season_role_growth_component_count"] >= 4,
            df["late_season_role_growth_component_count"].between(2, 3),
            df["late_season_role_growth_component_count"] == 1,
        ],
        ["high_multi_component", "medium_multi_component", "low_single_component"],
        default="unknown_no_component",
    )
    df["late_season_role_growth_note"] = df.apply(build_late_role_growth_note, axis=1)
    df["late_role_vs_full_season_gap"] = np.nan
    df["late_targets_vs_full_targets_gap"] = df["target_growth_component"]
    df["late_touches_vs_full_touches_gap"] = df["touch_growth_component"]
    df["late_snap_vs_full_snap_gap"] = df["snap_growth_component"]
    df["late_route_vs_full_route_gap"] = df["route_growth_component"]
    df["late_red_zone_role_growth"] = df["red_zone_growth_component"]
    df["late_role_growth_position_percentile"] = percentile_score_within_group(df, df["late_season_role_growth_score"], season_col, position_col)
    if adp_col:
        adp_pct = percentile_score_within_group(df, -safe_numeric(df[adp_col]), season_col, position_col)
        df["late_role_growth_adp_gap"] = df["late_role_growth_position_percentile"] - adp_pct
    else:
        df["late_role_growth_adp_gap"] = np.nan

    targets = detect_targets(df)
    score_dist = (
        df.groupby(position_col)["late_season_role_growth_score"]
        .agg(rows="size", valid_scores="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    bucket_counts = (
        df.groupby([position_col, "late_season_role_growth_bucket"])
        .size()
        .reset_index(name="rows")
        .sort_values([position_col, "late_season_role_growth_bucket"])
    )

    df.to_csv(OUTPUT, index=False)

    full_validation, predictions = run_walk_forward_validation(df, targets, season_col, position_col, adp_col, age_col)
    bucket_validation = run_bucket_validation(df, targets, position_col)
    window_validation = run_draft_window_validation(df, targets, season_col, position_col, adp_col, age_col)
    validation = pd.concat([bucket_validation, full_validation, window_validation], ignore_index=True, sort=False)
    classification, class_reason = classify_signal_strength(validation)
    validation["final_classification"] = classification
    validation.to_csv(VALIDATION_OUTPUT, index=False)

    concerns = []
    if not used.get("touch_growth_component"):
        concerns.append("No RB-specific touch/carry late-season growth columns were detected, so RB validation is coverage-limited.")
    if not used.get("route_growth_component"):
        concerns.append("No route participation growth columns were detected.")
    if not used.get("snap_growth_component"):
        concerns.append("No snap-share growth columns were detected.")
    if df["late_season_role_growth_score"].notna().mean() < 0.25:
        concerns.append("Late-season role growth score coverage is low across the full dataset.")
    concerns.append("The score uses only prior-season opportunity-growth columns and excludes all outcome/label columns.")

    diagnostics = {
        "rows_loaded": int(len(df)),
        "rows_saved": int(len(df)),
        "seasons_covered": [int(x) for x in sorted(safe_numeric(df[season_col]).dropna().unique())],
        "positions_covered": sorted(df[position_col].dropna().unique().tolist()),
        "season_col": season_col,
        "player_col": player_col,
        "position_col": position_col,
        "adp_col": adp_col,
        "positional_adp_col": positional_adp_col,
        "age_col": age_col,
        "source_columns_used": used,
        "source_columns_skipped": skipped,
        "components_created": component_cols,
        "valid_late_season_role_growth_scores": int(df["late_season_role_growth_score"].notna().sum()),
        "score_distribution_by_position": score_dist,
        "bucket_counts_by_position": bucket_counts,
        "validation_targets_available": targets,
        "classification": classification,
        "classification_reason": class_reason,
    }
    write_feature_report(df, input_path, diagnostics, component_cols)
    write_validation_report(validation, classification, class_reason, targets, concerns)

    best_full = validation[
        (validation.get("result_type") == "full_pool")
        & (validation.get("status") == "evaluated")
    ].copy()
    if not best_full.empty and "improvement_over_adp_only" in best_full.columns:
        best_full = best_full.sort_values("improvement_over_adp_only", ascending=False).head(1)
    best_window = validation[
        (validation.get("result_type") == "draft_window")
        & (validation.get("status") == "evaluated")
    ].copy()
    if not best_window.empty and "improvement_over_adp_only" in best_window.columns:
        best_window = best_window.sort_values("improvement_over_adp_only", ascending=False).head(1)

    preview_cols = [
        player_col,
        season_col,
        position_col,
        adp_col,
        age_col,
        "late_season_role_growth_score",
        "late_season_role_growth_bucket",
        "late_season_role_growth_note",
    ]
    preview_cols = [c for c in preview_cols if c]

    print(f"rows loaded: {len(df)}")
    print(f"rows saved: {len(df)}")
    print(f"seasons covered: {min(diagnostics['seasons_covered'])}-{max(diagnostics['seasons_covered'])}")
    print(f"positions covered: {', '.join(diagnostics['positions_covered'])}")
    print(f"detected player column: {player_col}")
    print(f"detected season column: {season_col}")
    print(f"detected position column: {position_col}")
    print(f"detected ADP column: {adp_col}")
    print(f"detected age score column: {age_col}")
    print("source columns used:")
    for component, cols in used.items():
        print(f"  {component}: {', '.join(cols) if cols else 'none'}")
    print("source columns skipped:")
    for component, cols in skipped.items():
        print(f"  {component}: {', '.join(cols) if cols else 'none'}")
    print(f"components created: {', '.join(component_cols)}")
    print("missingness rate for each component:")
    for col in component_cols + ["late_season_role_growth_score"]:
        print(f"  {col}: {df[col].isna().mean():.1%}")
    print("bucket counts by position:")
    print(bucket_counts.to_string(index=False))
    print(f"valid late_season_role_growth_score rows: {df['late_season_role_growth_score'].notna().sum()}")
    print("first 20 scored rows:")
    print(df[df["late_season_role_growth_score"].notna()][preview_cols].head(20).to_string(index=False))
    print(f"validation targets available: {', '.join(targets)}")
    print("best full-pool validation result:")
    print(best_full.to_string(index=False) if not best_full.empty else "none")
    print("best draft-window validation result:")
    print(best_window.to_string(index=False) if not best_window.empty else "none")
    late_eval = validation[(validation.get("model_name") == "ADP + late-season role growth") & (validation.get("status") == "evaluated")]
    combo_eval = validation[(validation.get("model_name") == "ADP + age + late-season role growth") & (validation.get("status") == "evaluated")]
    late_beat = bool((safe_numeric(late_eval.get("improvement_over_adp_only", pd.Series(dtype=float))) > 0).any()) if not late_eval.empty else False
    combo_beat = bool((safe_numeric(combo_eval.get("improvement_over_adp_only", pd.Series(dtype=float))) > 0).any()) if not combo_eval.empty else False
    print(f"ADP + late-season role growth beat ADP-only: {late_beat}")
    print(f"ADP + age + late-season role growth improved over ADP-only: {combo_beat}")
    print(f"final classification: {classification}")
    print(f"feature report: {FEATURE_REPORT}")
    print(f"validation report: {VALIDATION_REPORT}")
    print(f"output dataset: {OUTPUT}")
    print(f"validation CSV: {VALIDATION_OUTPUT}")


if __name__ == "__main__":
    main()






