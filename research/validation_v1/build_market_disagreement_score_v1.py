from __future__ import annotations

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
OUTPUT = VALIDATION_DIR / "predraft_validation_dataset_market_disagreement_score.csv"
FEATURE_REPORT = VALIDATION_DIR / "market_disagreement_score_v1_report.md"
VALIDATION_OUTPUT = VALIDATION_DIR / "market_disagreement_score_v1_validation.csv"
VALIDATION_REPORT = VALIDATION_DIR / "market_disagreement_score_v1_validation_report.md"

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

WR_TE_PROJECTION_CANDIDATES = [
    "projected_receptions",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_receiving_role_score",
    "projected_volume_score",
    "projected_total_tds",
]

RB_PROJECTION_CANDIDATES = [
    "projected_carries",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_touch_score",
    "projected_volume_score",
    "projected_total_tds",
    "projected_receptions",
]

ADP_CANDIDATES = ["overall_adp", "adp", "preseason_adp", "adp_rank"]
POSITIONAL_ADP_CANDIDATES = ["positional_adp", "pos_adp", "positional_adp_rank"]
UNSAFE_TARGET_PATTERNS = re.compile(r"beat_adp|underpriced|breakout|tier_jump|top12|top24|final_|finish", re.I)


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_position(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"WR", "RB", "TE", "QB"} else text


def detect_targets(df: pd.DataFrame) -> list[str]:
    found = [target for target in PREFERRED_TARGETS if target in df.columns]
    feature_prefixes = ("age_curve_", "market_disagreement_", "late_season_role_growth", "late_role_")
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


def projection_columns_for_position(position: str, df: pd.DataFrame) -> list[str]:
    candidates = RB_PROJECTION_CANDIDATES if position == "RB" else WR_TE_PROJECTION_CANDIDATES
    return [col for col in candidates if col in df.columns and safe_numeric(df[col]).notna().any()]


def rank_gap_component(df: pd.DataFrame, projection_col: str, adp_col: str, season_col: str, position_col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, _), group in df.groupby([season_col, position_col], dropna=False):
        proj = safe_numeric(group[projection_col])
        adp = safe_numeric(group[adp_col])
        valid = proj.notna() & adp.notna()
        if valid.sum() < 3:
            continue
        projection_rank = proj[valid].rank(ascending=False, method="average")
        adp_rank = adp[valid].rank(ascending=True, method="average")
        gap = adp_rank - projection_rank
        scale = max(float(valid.sum() - 1), 1.0)
        score = (50.0 + (gap / scale) * 50.0).clip(0, 100)
        out.loc[group[valid].index] = score
    return out


def assign_bucket(score: Any) -> str:
    if pd.isna(score):
        return "Unknown"
    score = float(score)
    if score >= 85:
        return "Major Positive Market Disagreement"
    if score >= 70:
        return "Positive Market Disagreement"
    if score >= 45:
        return "Neutral / Fairly Priced"
    if score >= 25:
        return "Negative Market Disagreement"
    return "Major Negative Market Disagreement"


def build_note(row: pd.Series) -> str:
    if pd.isna(row["market_disagreement_score"]):
        return "No usable preseason projection and ADP comparison was available."
    return f"{row['market_disagreement_bucket']} from {int(row['market_disagreement_component_count'])} projection-vs-ADP component(s)."


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


def top_decile_hit_rate(frame: pd.DataFrame, score_col: str, target: str) -> float:
    if frame.empty:
        return np.nan
    scored = frame[[score_col, target]].copy()
    scored[score_col] = safe_numeric(scored[score_col])
    scored[target] = safe_numeric(scored[target])
    scored = scored.dropna()
    if scored.empty:
        return np.nan
    top_n = max(1, int(math.ceil(len(scored) * 0.10)))
    return float(scored.sort_values(score_col, ascending=False).head(top_n)[target].mean())


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str]) -> tuple[pd.Series, str]:
    missing = [f for f in features if f not in train.columns or not train[f].notna().any() or not test[f].notna().any()]
    if missing:
        return pd.Series(np.nan, index=test.index), "skipped_required_feature_missing:" + ",".join(missing)
    y = safe_numeric(train[target])
    if y.nunique(dropna=True) < 2:
        return pd.Series(np.nan, index=test.index), "skipped_one_class_train"
    train_x = train[features].apply(pd.to_numeric, errors="coerce")
    test_x = test[features].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(numeric_only=True).fillna(0.0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    if LogisticRegression is None or Pipeline is None or StandardScaler is None:
        return test_x.rank(pct=True).mean(axis=1), "fallback_rank_score_no_sklearn"
    try:
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ])
        estimator.fit(train_x, y)
        return pd.Series(estimator.predict_proba(test_x)[:, 1], index=test.index), "fit"
    except Exception as exc:
        return test_x.rank(pct=True).mean(axis=1), f"fallback_rank_score_error:{exc.__class__.__name__}"


def model_groups(adp_col: str | None, age_col: str | None) -> dict[str, list[str]]:
    groups = {
        "ADP-only": [adp_col] if adp_col else [],
        "market disagreement only": ["market_disagreement_score"],
        "ADP + market disagreement": ([adp_col] if adp_col else []) + ["market_disagreement_score"],
    }
    if age_col:
        groups["ADP + age"] = ([adp_col] if adp_col else []) + [age_col]
        groups["ADP + age + market disagreement"] = ([adp_col] if adp_col else []) + [age_col, "market_disagreement_score"]
    return {k: [f for f in v if f] for k, v in groups.items() if v}


def summarize_predictions(predictions: pd.DataFrame, target: str, model_name: str, position: str | None, scope: str, window: str = "") -> dict[str, Any]:
    model_rows = predictions[predictions["model_name"] == model_name].copy()
    if model_rows.empty or model_rows["prediction"].notna().sum() == 0:
        return {"result_type": scope, "draft_window": window, "target": target, "position": position or "ALL", "model_name": model_name, "rows": int(len(model_rows)), "status": "skipped_no_valid_predictions"}
    adp_rows = predictions[predictions["model_name"] == "ADP-only"].copy()
    yearly_lifts = []
    for season, season_rows in model_rows.groupby("test_season"):
        base = adp_rows[adp_rows["test_season"] == season]
        model_hit = top_decile_hit_rate(season_rows, "prediction", target)
        adp_hit = top_decile_hit_rate(base, "prediction", target)
        if pd.notna(model_hit) and pd.notna(adp_hit):
            yearly_lifts.append(model_hit - adp_hit)
    model_hit = top_decile_hit_rate(model_rows, "prediction", target)
    adp_hit = top_decile_hit_rate(adp_rows, "prediction", target)
    improvement = model_hit - adp_hit if model_name != "ADP-only" and pd.notna(model_hit) and pd.notna(adp_hit) else 0.0
    return {
        "result_type": scope,
        "draft_window": window,
        "target": target,
        "position": position or "ALL",
        "model_name": model_name,
        "rows": int(len(model_rows)),
        "positive_rate": float(safe_numeric(model_rows[target]).mean()),
        "auc": auc_or_nan(model_rows[target], model_rows["prediction"]),
        "top_decile_hit_rate": model_hit,
        "baseline_hit_rate": adp_hit,
        "lift_over_baseline": model_hit - adp_hit if pd.notna(model_hit) and pd.notna(adp_hit) else np.nan,
        "improvement_over_adp_only": improvement,
        "seasons_tested": int(model_rows["test_season"].nunique()),
        "seasons_where_model_beat_adp_only": int(sum(1 for x in yearly_lifts if x > 0)) if model_name != "ADP-only" else 0,
        "average_yearly_lift": float(np.mean(yearly_lifts)) if yearly_lifts else np.nan,
        "median_yearly_lift": float(np.median(yearly_lifts)) if yearly_lifts else np.nan,
        "status": "evaluated",
    }


def walk_forward_predictions(df: pd.DataFrame, target: str, groups: dict[str, list[str]], season_col: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    skipped = []
    seasons = sorted(int(s) for s in safe_numeric(df[season_col]).dropna().unique())
    for test_season in seasons:
        train = df[safe_numeric(df[season_col]) < test_season].copy()
        test = df[safe_numeric(df[season_col]) == test_season].copy()
        train_y = safe_numeric(train[target])
        test_y = safe_numeric(test[target])
        if len(train) < 50 or train_y.sum() < 5 or (train_y.notna().sum() - train_y.sum()) < 5:
            skipped.append({"target": target, "test_season": test_season, "reason": "insufficient_training_rows_or_classes"})
            continue
        if len(test) < 20 or test_y.sum() < 2:
            skipped.append({"target": target, "test_season": test_season, "reason": "insufficient_test_rows_or_positives"})
            continue
        for model_name, features in groups.items():
            pred, status = fit_predict(train, test, target, features)
            part = test.copy()
            part["prediction"] = pred
            part["model_name"] = model_name
            part["test_season"] = test_season
            part["fit_status"] = status
            rows.append(part)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), skipped)


def run_validation(df: pd.DataFrame, targets: list[str], season_col: str, position_col: str, adp_col: str | None, age_col: str | None, scope: str = "full_pool", window: str = "") -> pd.DataFrame:
    rows = []
    groups = model_groups(adp_col, age_col)
    for target in targets:
        target_pos = infer_target_position(target)
        base = df[df[position_col].map(normalize_position) == target_pos].copy() if target_pos else df.copy()
        keep = []
        for candidate in sorted(set(sum(groups.values(), []) + [target, season_col, position_col, "player_name"])):
            if candidate in base.columns and candidate not in keep:
                keep.append(candidate)
        base = base[keep].copy()
        for col in sorted(set(sum(groups.values(), []) + [target, season_col])):
            if col in base.columns:
                base[col] = safe_numeric(base[col])
        base = base[base[target].notna()].copy()
        if len(base) < 100 and scope == "full_pool":
            rows.append({"result_type": scope, "draft_window": window, "target": target, "position": target_pos or "ALL", "status": "skipped_insufficient_full_pool_rows", "rows": len(base)})
            continue
        preds, skipped = walk_forward_predictions(base, target, groups, season_col)
        for item in skipped:
            item.update({"result_type": "season_skip", "draft_window": window, "position": target_pos or "ALL", "status": "skipped"})
            rows.append(item)
        if preds.empty:
            rows.append({"result_type": scope, "draft_window": window, "target": target, "position": target_pos or "ALL", "status": "skipped_no_walk_forward_predictions", "rows": len(base)})
            continue
        for model_name in groups:
            rows.append(summarize_predictions(preds, target, model_name, target_pos, scope, window))
    return pd.DataFrame(rows)


def run_draft_windows(df: pd.DataFrame, targets: list[str], season_col: str, position_col: str, adp_col: str | None, age_col: str | None) -> pd.DataFrame:
    if not adp_col:
        return pd.DataFrame([{"result_type": "draft_window", "status": "skipped_no_adp_column"}])
    rows = []
    adp = safe_numeric(df[adp_col])
    for label, low, high in ADP_WINDOWS:
        subset = df[adp >= low].copy() if math.isinf(high) else df[(adp >= low) & (adp <= high)].copy()
        if len(subset) < 40:
            rows.append({"result_type": "draft_window", "draft_window": label, "status": "skipped_insufficient_window_rows", "rows": len(subset)})
            continue
        result = run_validation(subset, targets, season_col, position_col, adp_col, age_col, "draft_window", label)
        rows.extend(result.to_dict("records"))
    return pd.DataFrame(rows)


def run_bucket_validation(df: pd.DataFrame, targets: list[str], position_col: str) -> pd.DataFrame:
    rows = []
    for target in targets:
        pos = infer_target_position(target)
        base = df[df[position_col].map(normalize_position) == pos].copy() if pos else df.copy()
        base = base[base[target].notna()].copy()
        baseline = float(safe_numeric(base[target]).mean()) if len(base) else np.nan
        for bucket, group in base.groupby("market_disagreement_bucket", dropna=False):
            y = safe_numeric(group[target])
            rows.append({
                "result_type": "bucket",
                "target": target,
                "position": pos or "ALL",
                "bucket": bucket,
                "rows": int(len(group)),
                "positives": int(y.sum()) if y.notna().any() else 0,
                "positive_rate": float(y.mean()) if y.notna().any() else np.nan,
                "baseline_positive_rate": baseline,
                "lift_over_baseline": float(y.mean() - baseline) if y.notna().any() and pd.notna(baseline) else np.nan,
                "status": "evaluated",
            })
    return pd.DataFrame(rows)


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
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def classify(validation: pd.DataFrame) -> tuple[str, str]:
    evaluated = validation[(validation.get("status") == "evaluated") & (validation.get("result_type") == "full_pool")].copy()
    if evaluated.empty:
        return "No Signal", "No full-pool walk-forward result cleared sample rules."
    market = evaluated[evaluated["model_name"].isin(["ADP + market disagreement", "ADP + age + market disagreement"])].copy()
    market["improvement_over_adp_only"] = safe_numeric(market.get("improvement_over_adp_only", pd.Series(dtype=float)))
    market["seasons_where_model_beat_adp_only"] = safe_numeric(market.get("seasons_where_model_beat_adp_only", pd.Series(dtype=float)))
    positive = market[(market["improvement_over_adp_only"] > 0) & (market["seasons_where_model_beat_adp_only"] >= 2)]
    windows = validation[(validation.get("status") == "evaluated") & (validation.get("result_type") == "draft_window")].copy()
    windows["improvement_over_adp_only"] = safe_numeric(windows.get("improvement_over_adp_only", pd.Series(dtype=float)))
    repeat_windows = windows[
        windows["model_name"].isin(["ADP + market disagreement", "ADP + age + market disagreement"])
        & (windows["improvement_over_adp_only"] > 0)
        & (safe_numeric(windows.get("seasons_where_model_beat_adp_only", pd.Series(dtype=float))) >= 2)
    ]
    if positive.empty:
        return "No Signal", "ADP plus market disagreement did not repeatably improve over ADP-only."
    if repeat_windows.empty:
        return "Weak Research Signal", "Some full-pool improvement exists, but draft-window repeatability is weak."
    if positive["improvement_over_adp_only"].max() >= 0.08 and repeat_windows["improvement_over_adp_only"].max() >= 0.08:
        return "Tie-Breaker Only", "Market disagreement shows modest repeatable lift, but not enough for app readiness."
    return "Weak Research Signal", "Positive evidence exists, but the effect size or repeatability is limited."


def write_reports(df: pd.DataFrame, validation: pd.DataFrame, diagnostics: dict[str, Any], classification: str, reason: str) -> None:
    dist = df.groupby("position")["market_disagreement_score"].agg(rows="size", valid_scores="count", mean="mean", median="median", min="min", max="max").reset_index()
    buckets = df.groupby(["position", "market_disagreement_bucket"]).size().reset_index(name="rows")
    best_full = validation[(validation.get("status") == "evaluated") & (validation.get("result_type") == "full_pool")].sort_values("improvement_over_adp_only", ascending=False).head(10)
    best_window = validation[(validation.get("status") == "evaluated") & (validation.get("result_type") == "draft_window")].sort_values("improvement_over_adp_only", ascending=False).head(10)
    feature_lines = [
        "# Market Disagreement Score V1",
        "",
        "## Executive summary",
        "Market Disagreement Score V1 compares archived preseason FantasyPros projection volume/rank signals against historical ADP inside each season-position group.",
        "",
        "This is research-only and was not promoted into the app.",
        "",
        "## Inputs and outputs",
        f"- Input: `{diagnostics['input_path']}`",
        f"- Output: `{OUTPUT}`",
        f"- Validation CSV: `{VALIDATION_OUTPUT}`",
        "",
        "## Columns detected",
        f"- ADP column used: `{diagnostics['adp_col']}`",
        f"- Positional ADP column available: `{diagnostics['positional_adp_col']}`",
        f"- Age score column: `{diagnostics['age_col']}`",
        "",
        "## Projection columns used",
    ]
    for pos, cols in diagnostics["projection_columns_used"].items():
        feature_lines.append(f"- {pos}: {', '.join(f'`{c}`' for c in cols) if cols else 'none'}")
    feature_lines.extend([
        "",
        "## Projection columns skipped",
    ])
    for pos, cols in diagnostics["projection_columns_skipped"].items():
        feature_lines.append(f"- {pos}: {', '.join(f'`{c}`' for c in cols) if cols else 'none'}")
    feature_lines.extend([
        "",
        "## Score distribution by position",
        markdown_table(dist),
        "",
        "## Bucket counts",
        markdown_table(buckets),
        "",
        "## Data concerns",
    ])
    feature_lines.extend(f"- {x}" for x in diagnostics["data_concerns"])
    FEATURE_REPORT.write_text("\n".join(feature_lines), encoding="utf-8")

    validation_lines = [
        "# Market Disagreement Score V1 Validation",
        "",
        "## Executive summary",
        f"Final classification: **{classification}**.",
        "",
        reason,
        "",
        "This is research-only. No app-readiness is claimed.",
        "",
        "## Targets tested",
        "\n".join(f"- `{target}`" for target in diagnostics["targets"]),
        "",
        "## Best full-pool validation results",
        markdown_table(best_full),
        "",
        "## Best draft-window validation results",
        markdown_table(best_window),
        "",
        "## ADP + market disagreement",
        markdown_table(validation[(validation.get("model_name") == "ADP + market disagreement") & (validation.get("status") == "evaluated")].sort_values("improvement_over_adp_only", ascending=False).head(10)),
        "",
        "## ADP + age + market disagreement",
        markdown_table(validation[(validation.get("model_name") == "ADP + age + market disagreement") & (validation.get("status") == "evaluated")].sort_values("improvement_over_adp_only", ascending=False).head(10)),
        "",
        "## Data concerns",
    ]
    validation_lines.extend(f"- {x}" for x in diagnostics["data_concerns"])
    validation_lines.extend([
        "",
        "## Recommended next step",
        "Use this as the main projection-vs-market research branch: inspect which projection components drive lift, then retest in draft windows and by positional ADP bucket before any app integration.",
    ])
    VALIDATION_REPORT.write_text("\n".join(validation_lines), encoding="utf-8")


def main() -> None:
    input_path = INPUT_PRIMARY if INPUT_PRIMARY.exists() else INPUT_FALLBACK
    if not input_path.exists():
        raise FileNotFoundError(f"No input found at {INPUT_PRIMARY} or {INPUT_FALLBACK}")
    df = pd.read_csv(input_path, low_memory=False)
    season_col = detect_column(df, ["season"])
    player_col = detect_column(df, ["player_name", "player", "name"])
    position_col = detect_column(df, ["position", "pos"])
    adp_col = detect_column(df, ADP_CANDIDATES)
    positional_adp_col = detect_column(df, POSITIONAL_ADP_CANDIDATES)
    age_col = detect_column(df, ["age_curve_edge_score", "age_curve_score"])
    if not season_col or not position_col or not adp_col:
        raise ValueError("Dataset must include season, position, and ADP columns.")
    df[position_col] = df[position_col].map(normalize_position)

    component_cols = []
    used_by_pos = {}
    skipped_by_pos = {}
    all_candidates = {"WR": WR_TE_PROJECTION_CANDIDATES, "RB": RB_PROJECTION_CANDIDATES, "TE": WR_TE_PROJECTION_CANDIDATES}
    for pos, candidates in all_candidates.items():
        used = projection_columns_for_position(pos, df)
        used_by_pos[pos] = used
        skipped_by_pos[pos] = [c for c in candidates if c not in used]
        for col in used:
            comp = f"market_disagreement_component_{col}"
            if comp not in df.columns:
                df[comp] = np.nan
            mask = df[position_col] == pos
            df.loc[mask, comp] = rank_gap_component(df[mask], col, adp_col, season_col, position_col)
            component_cols.append(comp)
    component_cols = sorted(set(component_cols))
    df["market_disagreement_component_count"] = df[component_cols].notna().sum(axis=1) if component_cols else 0
    df["market_disagreement_score"] = df[component_cols].mean(axis=1, skipna=True) if component_cols else np.nan
    df.loc[df["market_disagreement_component_count"] == 0, "market_disagreement_score"] = np.nan
    df["market_disagreement_bucket"] = df["market_disagreement_score"].map(assign_bucket)
    df["market_disagreement_note"] = df.apply(build_note, axis=1)
    df.to_csv(OUTPUT, index=False)

    targets = detect_targets(df)
    full = run_validation(df, targets, season_col, position_col, adp_col, age_col)
    buckets = run_bucket_validation(df, targets, position_col)
    windows = run_draft_windows(df, targets, season_col, position_col, adp_col, age_col)
    validation = pd.concat([buckets, full, windows], ignore_index=True, sort=False)
    classification, reason = classify(validation)
    validation["final_classification"] = classification
    validation.to_csv(VALIDATION_OUTPUT, index=False)

    concerns = []
    if df["market_disagreement_score"].notna().mean() < 0.25:
        concerns.append("Market disagreement coverage is low; projection data only exists for imported FantasyPros Wayback seasons.")
    if not used_by_pos.get("RB"):
        concerns.append("No usable RB projection components were detected.")
    if not used_by_pos.get("WR"):
        concerns.append("No usable WR projection components were detected.")
    concerns.append("The score uses preseason projections and ADP only; final outcomes and target labels are excluded from score construction.")
    diagnostics = {
        "input_path": str(input_path),
        "rows_loaded": int(len(df)),
        "rows_saved": int(len(df)),
        "seasons": [int(x) for x in sorted(safe_numeric(df[season_col]).dropna().unique())],
        "positions": sorted(df[position_col].dropna().unique().tolist()),
        "adp_col": adp_col,
        "positional_adp_col": positional_adp_col,
        "age_col": age_col,
        "projection_columns_used": used_by_pos,
        "projection_columns_skipped": skipped_by_pos,
        "valid_scores": int(df["market_disagreement_score"].notna().sum()),
        "targets": targets,
        "data_concerns": concerns,
    }
    write_reports(df, validation, diagnostics, classification, reason)

    best_full = validation[(validation.get("status") == "evaluated") & (validation.get("result_type") == "full_pool")].sort_values("improvement_over_adp_only", ascending=False).head(1)
    best_window = validation[(validation.get("status") == "evaluated") & (validation.get("result_type") == "draft_window")].sort_values("improvement_over_adp_only", ascending=False).head(1)
    preview_cols = [c for c in [player_col, season_col, position_col, adp_col, age_col, "market_disagreement_score", "market_disagreement_bucket", "market_disagreement_note"] if c]
    print(f"rows loaded: {len(df)}")
    print(f"rows saved: {len(df)}")
    print(f"seasons covered: {min(diagnostics['seasons'])}-{max(diagnostics['seasons'])}")
    print(f"positions covered: {', '.join(diagnostics['positions'])}")
    print(f"ADP column used: {adp_col}")
    print("projection columns used:")
    for pos, cols in used_by_pos.items():
        print(f"  {pos}: {', '.join(cols) if cols else 'none'}")
    print("projection columns skipped:")
    for pos, cols in skipped_by_pos.items():
        print(f"  {pos}: {', '.join(cols) if cols else 'none'}")
    print(f"valid market disagreement scores: {diagnostics['valid_scores']}")
    print("first 20 scored rows:")
    print(df[df["market_disagreement_score"].notna()][preview_cols].head(20).to_string(index=False))
    print(f"available targets: {', '.join(targets)}")
    print("best full-pool validation result:")
    print(best_full.to_string(index=False) if not best_full.empty else "none")
    print("best draft-window validation result:")
    print(best_window.to_string(index=False) if not best_window.empty else "none")
    market_eval = validation[(validation.get("model_name") == "ADP + market disagreement") & (validation.get("status") == "evaluated")]
    age_market_eval = validation[(validation.get("model_name") == "ADP + age + market disagreement") & (validation.get("status") == "evaluated")]
    market_beat = bool((safe_numeric(market_eval.get("improvement_over_adp_only", pd.Series(dtype=float))) > 0).any()) if not market_eval.empty else False
    age_helped = bool((safe_numeric(age_market_eval.get("improvement_over_adp_only", pd.Series(dtype=float))) > 0).any()) if not age_market_eval.empty else False
    print(f"ADP + market disagreement beat ADP-only: {market_beat}")
    print(f"adding age helped: {age_helped}")
    print(f"final classification: {classification}")
    print(f"feature report: {FEATURE_REPORT}")
    print(f"validation report: {VALIDATION_REPORT}")
    print(f"output dataset: {OUTPUT}")
    print(f"validation CSV: {VALIDATION_OUTPUT}")


if __name__ == "__main__":
    main()
