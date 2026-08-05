from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None
    RandomForestClassifier = None
    Pipeline = None
    StandardScaler = None
    roc_auc_score = None


PROJECT_ROOT = Path(r"C:\Users\cklut\Desktop\Projects\Guaranteed Play")
VALIDATION_DIR = PROJECT_ROOT / "research" / "validation_v1"
CASE_STUDIES_DIR = PROJECT_ROOT / "case_studies"
DASHBOARD = CASE_STUDIES_DIR / "output" / "fantasy_age_study_dashboard.html"
INPUT = VALIDATION_DIR / "predraft_validation_dataset_projected.csv"
OUTPUT = VALIDATION_DIR / "predraft_validation_dataset_age_edge_score.csv"
FEATURE_REPORT = VALIDATION_DIR / "age_curve_edge_score_v1_report.md"
VALIDATION_OUTPUT = VALIDATION_DIR / "age_curve_score_v1_validation.csv"
VALIDATION_REPORT = VALIDATION_DIR / "age_curve_score_v1_validation_report.md"

RATE_FILES = {
    "RB": CASE_STUDIES_DIR / "output" / "rb_elite_age_rates.csv",
    "WR": CASE_STUDIES_DIR / "output" / "wr_elite_age_rates.csv",
    "TE": CASE_STUDIES_DIR / "output" / "te_elite_age_rates.csv",
    "QB": CASE_STUDIES_DIR / "output" / "qb_elite_age_rates.csv",
}

FALLBACK_CURVES = {
    "RB": {20: 70, 21: 85, 22: 100, 23: 100, 24: 96, 25: 90, 26: 76, 27: 58, 28: 40, 29: 25, 30: 15},
    "WR": {20: 65, 21: 78, 22: 86, 23: 92, 24: 100, 25: 100, 26: 97, 27: 93, 28: 85, 29: 72, 30: 55, 31: 38, 32: 25},
    "TE": {21: 55, 22: 65, 23: 76, 24: 86, 25: 95, 26: 100, 27: 100, 28: 96, 29: 90, 30: 78, 31: 65, 32: 50, 33: 35},
    "QB": {21: 60, 22: 70, 23: 78, 24: 84, 25: 88, 26: 92, 27: 96, 28: 100, 29: 100, 30: 100, 31: 98, 32: 96, 33: 93, 34: 88, 35: 80, 36: 70, 37: 58, 38: 45, 39: 30},
}

ADP_WINDOWS = [
    ("1-24", 1, 24),
    ("25-48", 25, 48),
    ("49-72", 49, 72),
    ("73-96", 73, 96),
    ("97-120", 97, 120),
    ("121-150", 121, 150),
    ("151+", 151, math.inf),
]

PREFERRED_TARGETS = [
    "WR_Beat_ADP_By_12",
    "RB_Beat_ADP_By_12",
    "WR_Underpriced_Top24",
    "RB_Underpriced_Top24",
    "WR_Underpriced_Top12",
    "RB_Underpriced_Top12",
]


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_position(value: Any) -> str:
    text = str(value or "").upper().strip()
    if text in {"RB", "WR", "TE", "QB"}:
        return text
    return text


def find_related_age_files() -> list[str]:
    patterns = re.compile(r"age|age_curve|fantasy_age|aging|peak|breakout|position_age|rb_age|wr_age|te_age|qb_age", re.I)
    allowed = {".csv", ".parquet", ".json", ".md", ".txt", ".py", ".html"}
    files = []
    for path in CASE_STUDIES_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed and patterns.search(path.name):
            files.append(str(path))
    return sorted(files)


def parse_age_dashboard() -> dict[str, Any]:
    result = {
        "dashboard_found": DASHBOARD.exists(),
        "dashboard_parsed_successfully": False,
        "dashboard_path": str(DASHBOARD),
        "html_length": 0,
        "text_findings": [],
        "tables_found": 0,
        "plotly_found": False,
    }
    if not DASHBOARD.exists():
        return result
    html = DASHBOARD.read_text(encoding="utf-8", errors="replace")
    result["html_length"] = len(html)
    result["plotly_found"] = "Plotly.newPlot" in html or "plotly" in html.lower()
    stripped = re.sub(r"<[^>]+>", " ", html)
    stripped = re.sub(r"\s+", " ", stripped)
    snippets = []
    for term in ["RB", "WR", "TE", "QB", "Top", "Elite", "age"]:
        idx = stripped.lower().find(term.lower())
        if idx >= 0:
            snippets.append(stripped[max(0, idx - 120) : idx + 220])
    result["text_findings"] = snippets[:10]
    try:
        tables = pd.read_html(str(DASHBOARD))
        result["tables_found"] = len(tables)
    except Exception:
        result["tables_found"] = 0
    result["dashboard_parsed_successfully"] = bool(result["html_length"] > 0 and (snippets or result["tables_found"] >= 0))
    return result


def normalize_0_100(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() == 0:
        return pd.Series(np.nan, index=v.index)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return pd.Series(50.0, index=v.index)
    return (v - lo) / (hi - lo) * 100


def build_verified_age_score_map() -> tuple[dict[str, dict[int, float]], dict[str, pd.DataFrame], list[str]]:
    maps: dict[str, dict[int, float]] = {}
    frames: dict[str, pd.DataFrame] = {}
    used_files = []
    metric_weights = [
        ("top12_rate", 0.40),
        ("top24_rate", 0.30),
        ("top36_rate", 0.20),
        ("top6_rate", 0.10),
    ]
    for pos, path in RATE_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Age" not in df.columns:
            continue
        score_parts = []
        weights = []
        for col, weight in metric_weights:
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
                score_parts.append(normalize_0_100(df[col]) * weight)
                weights.append(weight)
        if not score_parts:
            continue
        score = sum(score_parts) / sum(weights)
        count_col = next((c for c in df.columns if c.startswith("total_") and c.endswith("_seasons")), None)
        if count_col:
            count = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
            penalty = np.where(count < 20, 0.75, np.where(count < 50, 0.90, 1.0))
            score = score * penalty
        df = df.copy()
        df["age_curve_edge_score"] = score.clip(0, 100)
        maps[pos] = {int(row["Age"]): float(row["age_curve_edge_score"]) for _, row in df.iterrows()}
        frames[pos] = df
        used_files.append(str(path))
    return maps, frames, used_files


def fallback_score(pos: str, age: float) -> float:
    curve = FALLBACK_CURVES.get(pos)
    if curve is None or pd.isna(age):
        return np.nan
    age_int = int(math.floor(age))
    if pos == "RB" and age_int <= 20:
        return 70.0
    if pos == "WR" and age_int <= 20:
        return 65.0
    if pos == "TE" and age_int <= 21:
        return 55.0
    if pos == "QB" and age_int <= 21:
        return 60.0
    keys = sorted(curve)
    if age_int <= keys[0]:
        return float(curve[keys[0]])
    if age_int >= keys[-1]:
        return float(curve[keys[-1]])
    return float(curve.get(age_int, np.nan))


def score_age(pos: str, age: float, score_map: dict[str, dict[int, float]]) -> float:
    if pd.isna(age):
        return np.nan
    pos = normalize_position(pos)
    if pos not in score_map:
        return fallback_score(pos, age)
    age_int = int(math.floor(age))
    curve = score_map[pos]
    if age_int in curve:
        return curve[age_int]
    keys = sorted(curve)
    if not keys:
        return fallback_score(pos, age)
    if age_int <= keys[0]:
        return curve[keys[0]]
    if age_int >= keys[-1]:
        return curve[keys[-1]]
    # linear interpolation between neighboring observed ages
    lower = max(k for k in keys if k <= age_int)
    upper = min(k for k in keys if k >= age_int)
    if lower == upper:
        return curve[lower]
    return float((curve[lower] + curve[upper]) / 2)


def assign_bucket(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 90:
        return "Elite Age Window"
    if score >= 75:
        return "Strong Age Window"
    if score >= 55:
        return "Neutral Age Window"
    if score >= 35:
        return "Mild Age Risk"
    return "Major Age Risk"


def age_note(pos: str, age: float, score: float) -> str:
    if pd.isna(age) or pd.isna(score):
        return "Missing age or unsupported position; no age edge score."
    bucket = assign_bucket(score)
    return f"{pos} age {age:.1f}: {bucket} based on verified fantasy age-rate curves."


def add_age_score(df: pd.DataFrame, score_map: dict[str, dict[int, float]], source: str) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    cols = {
        "age": detect_column(out, ["age", "player_age", "draft_age", "season_age", "age_at_season_start"]),
        "position": detect_column(out, ["position", "pos", "fantasy_position"]),
        "season": detect_column(out, ["season", "year"]),
        "player": detect_column(out, ["player", "player_name", "name", "player_display_name"]),
        "adp": detect_column(out, ["adp", "overall_adp", "adp_rank", "draft_adp", "fantasypros_adp", "ffc_adp"]),
        "positional_adp": detect_column(out, ["positional_adp", "pos_adp", "positional_adp_rank", "position_adp", "position_rank_adp"]),
    }
    age = safe_numeric(out[cols["age"]]) if cols["age"] else pd.Series(np.nan, index=out.index)
    pos = out[cols["position"]].apply(normalize_position) if cols["position"] else pd.Series("", index=out.index)
    scores = [score_age(p, a, score_map) for p, a in zip(pos, age)]
    out["age_curve_edge_score"] = pd.Series(scores, index=out.index).clip(0, 100)
    out["age_curve_edge_bucket"] = out["age_curve_edge_score"].apply(assign_bucket)
    out["age_curve_edge_note"] = [age_note(p, a, s) for p, a, s in zip(pos, age, out["age_curve_edge_score"])]
    out["age_curve_source"] = source
    out["age_curve_component_confidence"] = np.where(out["age_curve_edge_score"].notna(), "medium_verified_age_rate_curve", "missing")
    peak_by_pos = {}
    for p, curve in score_map.items():
        if curve:
            best_age = max(curve, key=curve.get)
            peak_by_pos[p] = best_age
    out["age_curve_peak_distance"] = [abs(float(a) - peak_by_pos.get(p, np.nan)) if pd.notna(a) and p in peak_by_pos else np.nan for p, a in zip(pos, age)]
    out["age_curve_position_percentile"] = out.groupby(cols["position"])["age_curve_edge_score"].rank(pct=True) if cols["position"] else np.nan
    out["age_curve_risk_flag"] = out["age_curve_edge_bucket"].isin(["Mild Age Risk", "Major Age Risk"]).astype(int)
    out["age_curve_breakout_window_flag"] = out["age_curve_edge_bucket"].isin(["Elite Age Window", "Strong Age Window"]).astype(int)
    out["age_curve_decline_window_flag"] = out["age_curve_edge_bucket"].isin(["Mild Age Risk", "Major Age Risk"]).astype(int)
    return out, cols


def target_position(target: str) -> str | None:
    if target.startswith("WR_"):
        return "WR"
    if target.startswith("RB_"):
        return "RB"
    if target.startswith("TE_"):
        return "TE"
    if target.startswith("QB_"):
        return "QB"
    return None


def detect_targets(df: pd.DataFrame) -> list[str]:
    targets = [t for t in PREFERRED_TARGETS if t in df.columns]
    extra = [c for c in df.columns if any(term in c for term in ["Beat_ADP", "Underpriced", "Breakout", "Tier_Jump"]) and c not in targets]
    return targets + extra


def auc(y: pd.Series, s: pd.Series) -> float:
    if roc_auc_score is None:
        return np.nan
    mask = y.notna() & s.notna()
    if mask.sum() < 20 or y[mask].nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(y[mask], s[mask]))
    except Exception:
        return np.nan


def estimator(kind: str):
    if kind == "logistic" and LogisticRegression is not None:
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))])
    if kind == "random_forest" and RandomForestClassifier is not None:
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=10, random_state=42, class_weight="balanced_subsample")
    return None


def fit_scores(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], kind: str = "logistic") -> tuple[pd.Series, str]:
    available = [f for f in features if f in train.columns and train[f].notna().any()]
    if not available:
        return pd.Series(np.nan, index=test.index), "skipped_no_features"
    x_train = train[available].apply(pd.to_numeric, errors="coerce")
    x_test = test[available].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train[target], errors="coerce")
    med = x_train.median(numeric_only=True).fillna(0)
    x_train = x_train.fillna(med)
    x_test = x_test.fillna(med)
    clf = estimator(kind)
    if clf is None:
        return pd.Series(x_test.rank(pct=True).mean(axis=1), index=test.index), "rank_fallback"
    try:
        clf.fit(x_train, y_train)
        return pd.Series(clf.predict_proba(x_test)[:, 1], index=test.index), "fit"
    except Exception as exc:
        return pd.Series(np.nan, index=test.index), f"error_{exc.__class__.__name__}"


def top_decile_hit(test: pd.DataFrame, target: str, score: pd.Series) -> float:
    scored = test.copy()
    scored["_score"] = score
    scored = scored[scored["_score"].notna()]
    if scored.empty:
        return np.nan
    n = max(1, int(math.ceil(len(scored) * 0.10)))
    return float(pd.to_numeric(scored.sort_values("_score", ascending=False).head(n)[target], errors="coerce").mean())


def run_walk_forward(df: pd.DataFrame, targets: list[str], adp_col: str | None) -> pd.DataFrame:
    rows = []
    if adp_col is None:
        return pd.DataFrame(rows)
    for target in targets:
        pos = target_position(target)
        subset = df.copy()
        if pos:
            subset = subset[subset["position"].eq(pos)]
        subset = subset[pd.to_numeric(subset[target], errors="coerce").notna() & subset["age_curve_edge_score"].notna() & pd.to_numeric(subset[adp_col], errors="coerce").notna()].copy()
        subset["season"] = pd.to_numeric(subset["season"], errors="coerce")
        subset[target] = pd.to_numeric(subset[target], errors="coerce")
        if len(subset) < 100:
            rows.append({"validation_type": "walk_forward", "target": target, "position": pos or "ALL", "model": "all", "status": "skipped_full_pool_sample", "rows": len(subset)})
            continue
        season_rows = []
        for season in sorted(subset["season"].dropna().astype(int).unique()):
            train = subset[subset["season"] < season].copy()
            test = subset[subset["season"].eq(season)].copy()
            if len(train) < 50 or len(test) < 20 or train[target].sum() < 5 or (len(train) - train[target].sum()) < 5 or test[target].sum() < 2:
                continue
            model_defs = {
                "ADP-only": [adp_col],
                "ADP + age_curve_edge_score": [adp_col, "age_curve_edge_score"],
                "age_curve_edge_score only": ["age_curve_edge_score"],
            }
            season_metric = {"season": season, "rows": len(test), "baseline_rate": float(test[target].mean())}
            for model, features in model_defs.items():
                scores, status = fit_scores(train, test, target, features, "logistic")
                season_metric[f"{model}_hit"] = top_decile_hit(test, target, scores)
                season_metric[f"{model}_auc"] = auc(test[target], scores)
                season_metric[f"{model}_status"] = status
            season_rows.append(season_metric)
        if not season_rows:
            rows.append({"validation_type": "walk_forward", "target": target, "position": pos or "ALL", "model": "all", "status": "skipped_no_valid_seasons", "rows": len(subset)})
            continue
        sf = pd.DataFrame(season_rows)
        adp_hit = sf["ADP-only_hit"]
        for model in ["ADP-only", "ADP + age_curve_edge_score", "age_curve_edge_score only"]:
            hit = sf[f"{model}_hit"]
            lift = hit - sf["baseline_rate"]
            rows.append({
                "validation_type": "walk_forward",
                "target": target,
                "position": pos or "ALL",
                "model": model,
                "status": "completed",
                "rows": int(subset.shape[0]),
                "positive_rate": float(subset[target].mean()),
                "seasons_tested": int(sf["season"].nunique()),
                "auc": float(sf[f"{model}_auc"].mean()),
                "top_decile_hit_rate": float(hit.mean()),
                "baseline_hit_rate": float(sf["baseline_rate"].mean()),
                "lift_over_baseline": float(lift.mean()),
                "improvement_over_adp_only": float((hit - adp_hit).mean()) if model != "ADP-only" else 0.0,
                "seasons_adp_age_beats_adp_only": int(((sf["ADP + age_curve_edge_score_hit"] - sf["ADP-only_hit"]) > 0).sum()),
                "average_yearly_lift": float(lift.mean()),
                "median_yearly_lift": float(lift.median()),
            })
    return pd.DataFrame(rows)


def run_bucket_validation(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["age_score_decile"] = pd.qcut(work["age_curve_edge_score"], q=10, labels=False, duplicates="drop")
    for target in targets:
        pos = target_position(target)
        subset = work[work["position"].eq(pos)].copy() if pos else work.copy()
        subset = subset[pd.to_numeric(subset[target], errors="coerce").notna()]
        baseline = float(pd.to_numeric(subset[target], errors="coerce").mean()) if len(subset) else np.nan
        for group_col in ["age_curve_edge_bucket", "age_score_decile"]:
            for key, group in subset.groupby(group_col, dropna=False):
                if len(group) < 20:
                    continue
                rate = float(pd.to_numeric(group[target], errors="coerce").mean())
                rows.append({
                    "validation_type": "bucket",
                    "target": target,
                    "position": pos or "ALL",
                    "bucket_type": group_col,
                    "bucket": str(key),
                    "rows": int(len(group)),
                    "positive_count": int(pd.to_numeric(group[target], errors="coerce").sum()),
                    "positive_rate": rate,
                    "baseline_positive_rate": baseline,
                    "lift_over_baseline": rate - baseline if pd.notna(baseline) else np.nan,
                })
    return pd.DataFrame(rows)


def adp_window(value: object) -> str | None:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    for label, low, high in ADP_WINDOWS:
        if val >= low and val <= high:
            return label
    return None


def run_draft_window_validation(df: pd.DataFrame, targets: list[str], adp_col: str | None) -> pd.DataFrame:
    if adp_col is None:
        return pd.DataFrame()
    work = df.copy()
    work["adp_window"] = work[adp_col].apply(adp_window)
    rows = []
    for target in targets:
        pos = target_position(target)
        subset = work[work["position"].eq(pos)].copy() if pos else work.copy()
        subset = subset[pd.to_numeric(subset[target], errors="coerce").notna() & subset["age_curve_edge_score"].notna() & subset["adp_window"].notna()].copy()
        for window, group in subset.groupby("adp_window", dropna=True):
            if len(group) < 40:
                rows.append({"validation_type": "draft_window", "target": target, "position": pos or "ALL", "draft_window": window, "status": "skipped_small_sample", "rows": len(group)})
                continue
            season_rows = []
            for season in sorted(group["season"].dropna().astype(int).unique()):
                train = subset[subset["season"] < season].copy()
                test = group[group["season"].eq(season)].copy()
                if len(train) < 50 or len(test) < 10 or pd.to_numeric(train[target], errors="coerce").sum() < 5 or pd.to_numeric(test[target], errors="coerce").sum() < 1:
                    continue
                adp_scores, _ = fit_scores(train, test, target, [adp_col], "logistic")
                age_scores, _ = fit_scores(train, test, target, [adp_col, "age_curve_edge_score"], "logistic")
                adp_hit = top_decile_hit(test, target, adp_scores)
                age_hit = top_decile_hit(test, target, age_scores)
                season_rows.append({"season": season, "adp_hit": adp_hit, "age_hit": age_hit})
            if not season_rows:
                rows.append({"validation_type": "draft_window", "target": target, "position": pos or "ALL", "draft_window": window, "status": "skipped_no_valid_seasons", "rows": len(group)})
                continue
            sf = pd.DataFrame(season_rows)
            rows.append({
                "validation_type": "draft_window",
                "target": target,
                "position": pos or "ALL",
                "draft_window": window,
                "status": "completed",
                "rows": int(len(group)),
                "positive_rate": float(pd.to_numeric(group[target], errors="coerce").mean()),
                "adp_only_top_decile_hit_rate": float(sf["adp_hit"].mean()),
                "adp_age_top_decile_hit_rate": float(sf["age_hit"].mean()),
                "improvement_over_adp_only": float((sf["age_hit"] - sf["adp_hit"]).mean()),
                "seasons_tested": int(sf["season"].nunique()),
                "seasons_adp_age_beats_adp_only": int((sf["age_hit"] > sf["adp_hit"]).sum()),
            })
    return pd.DataFrame(rows)


def classify_signal(validation: pd.DataFrame, draft: pd.DataFrame) -> tuple[str, str, pd.Series | None, pd.Series | None]:
    wf = validation[(validation["validation_type"].eq("walk_forward")) & (validation["model"].eq("ADP + age_curve_edge_score")) & (validation["status"].eq("completed"))].copy()
    if wf.empty:
        return "No Signal", "Validation could not produce completed ADP + age walk-forward results.", None, None
    best = wf.sort_values(["improvement_over_adp_only", "seasons_adp_age_beats_adp_only", "seasons_tested"], ascending=[False, False, False]).iloc[0]
    draft_completed = draft[draft.get("status", pd.Series(dtype=str)).eq("completed")].copy() if not draft.empty else pd.DataFrame()
    best_draft = None
    if not draft_completed.empty:
        best_draft = draft_completed.sort_values(["improvement_over_adp_only", "seasons_adp_age_beats_adp_only", "rows"], ascending=[False, False, False]).iloc[0]
    if best["improvement_over_adp_only"] <= 0:
        return "No Signal", "ADP + age did not improve over ADP-only in the best full-pool walk-forward result.", best, best_draft
    if best["seasons_adp_age_beats_adp_only"] < max(2, best["seasons_tested"] * 0.4):
        return "Weak Research Signal", "Some improvement exists, but season repeatability is weak.", best, best_draft
    if best_draft is None or best_draft.get("improvement_over_adp_only", 0) <= 0:
        return "Weak Research Signal", "Full-pool improvement exists, but draft-window support is weak or missing.", best, best_draft
    if best["improvement_over_adp_only"] >= 0.05 and best_draft["improvement_over_adp_only"] >= 0.05:
        return "Tie-Breaker Only", "Age adds modest repeatable lift, but this remains a secondary signal and is not app-ready.", best, best_draft
    return "Weak Research Signal", "Age adds small lift, but not enough for a tie-breaker classification.", best, best_draft


def write_feature_report(meta: dict[str, Any], score_frames: dict[str, pd.DataFrame], cols: dict[str, str], df: pd.DataFrame) -> None:
    lines = [
        "# Age Curve Edge Score V1 Feature Report",
        "",
        "Research-only. This does not modify the app, Draft Mode, rankings, recommendations, player cards, or app-facing scores.",
        "",
        "## Executive Summary",
        "",
        f"Built `age_curve_edge_score` from verified age-study artifacts. Dashboard found: `{meta['dashboard']['dashboard_found']}`. Dashboard parsed: `{meta['dashboard']['dashboard_parsed_successfully']}`. Fallback curves needed: `{meta['fallback_needed']}`.",
        "",
        "## Input/Output Files",
        "",
        f"- Input: `{INPUT}`",
        f"- Output: `{OUTPUT}`",
        f"- Primary dashboard: `{DASHBOARD}`",
        "",
        "## Related Source Files Found",
        "",
        "\n".join(f"- `{p}`" for p in meta["related_files"][:40]) or "None.",
        "",
        "## Related Source Files Used",
        "",
        "\n".join(f"- `{p}`" for p in meta["used_files"]) or "None.",
        "",
        "## Extracted Age Findings",
        "",
    ]
    for pos, frame in score_frames.items():
        top = frame.sort_values("age_curve_edge_score", ascending=False).head(5)
        lines.append(f"- {pos}: strongest ages by normalized blended elite/useful finish rates: " + ", ".join(f"{int(r.Age)} ({r.age_curve_edge_score:.1f})" for _, r in top.iterrows()))
    lines.extend([
        "",
        "## Score Conversion",
        "",
        "For each position, age-level Top12, Top24, Top36, and Top6 rates were normalized within position to 0-100 and blended with weights 40%, 30%, 20%, and 10% when available. Small sample ages were penalized. Player-season scoring uses only age and position.",
        "",
        "## Columns Detected",
        "",
        json.dumps(cols, indent=2),
        "",
        "## Bucket Definitions",
        "",
        "- 90-100: Elite Age Window",
        "- 75-89: Strong Age Window",
        "- 55-74: Neutral Age Window",
        "- 35-54: Mild Age Risk",
        "- 0-34: Major Age Risk",
        "- NaN: Unknown",
        "",
        "## Missingness Diagnostics",
        "",
        f"Rows loaded/saved: `{len(df)}`. Valid age scores: `{int(df['age_curve_edge_score'].notna().sum())}`. Missing age scores: `{int(df['age_curve_edge_score'].isna().sum())}`.",
        "",
        "## Score Distribution By Position",
        "",
        df.groupby("position")["age_curve_edge_score"].agg(["count", "mean", "min", "max"]).reset_index().to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Bucket Counts By Position",
        "",
        df.groupby(["position", "age_curve_edge_bucket"]).size().reset_index(name="rows").to_markdown(index=False),
        "",
        "## Data Concerns",
        "",
        "- The score is calibrated from historical outcome rates in the age study, but player-season assignment uses only age and position.",
        "- This is a standalone feature, not an app-facing recommendation.",
        "- TE/QB curves are built, but validation targets in the current WR/RB dataset are mostly WR/RB specific.",
    ])
    FEATURE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_validation_report(validation: pd.DataFrame, bucket: pd.DataFrame, draft: pd.DataFrame, targets: list[str], classification: str, reason: str, best: pd.Series | None, best_draft: pd.Series | None) -> None:
    lines = [
        "# Age Curve Edge Score V1 Validation Report",
        "",
        "Research-only. Do not promote this score into the app; do not claim app-readiness.",
        "",
        "## Executive Summary",
        "",
        f"Final classification: **{classification}**. {reason}",
        "",
        "## Targets Tested",
        "",
        ", ".join(targets) if targets else "No targets available.",
        "",
        "## Validation Methodology",
        "",
        "Used season-based walk-forward logistic regression comparing ADP-only, ADP + age_curve_edge_score, and age-only. Also ran bucket-level target rates and ADP draft-window comparisons where sample rules allowed.",
        "",
        "## Bucket-Level Validation",
        "",
        bucket.head(30).to_markdown(index=False, floatfmt=".3f") if not bucket.empty else "No bucket validation rows.",
        "",
        "## ADP-Only vs ADP + Age Validation",
        "",
        validation[validation["validation_type"].eq("walk_forward")].sort_values(["improvement_over_adp_only"], ascending=False).head(30).to_markdown(index=False, floatfmt=".3f") if not validation.empty else "No walk-forward validation rows.",
        "",
        "## Draft-Window Validation",
        "",
        draft.sort_values(["improvement_over_adp_only"], ascending=False).head(30).to_markdown(index=False, floatfmt=".3f") if not draft.empty else "No draft-window validation rows.",
        "",
        "## Season Repeatability",
        "",
        "Repeatability is measured by seasons where ADP + age beats ADP-only in top-decile hit rate.",
        "",
        "## Best Result",
        "",
        best.to_json() if best is not None else "No completed full-pool best result.",
        "",
        "## Best Draft-Window Result",
        "",
        best_draft.to_json() if best_draft is not None else "No completed draft-window best result.",
        "",
        "## Whether Age Adds Independent Signal Beyond ADP",
        "",
        "This is answered by `improvement_over_adp_only`. Positive values indicate ADP + age beat ADP-only in the tested walk-forward setup.",
        "",
        "## Recommendation For Next Research Step",
        "",
        "If the score shows useful lift, test it inside combined-feature models alongside projections and role features. If weak, keep it as a diagnostic/risk feature only and do not app-integrate.",
    ]
    VALIDATION_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dashboard = parse_age_dashboard()
    related = find_related_age_files()
    score_map, score_frames, used_files = build_verified_age_score_map()
    fallback_needed = not bool(score_map)
    if fallback_needed:
        score_map = FALLBACK_CURVES
        source = "fallback_curve_no_verified_dashboard_data"
    else:
        source = "fantasy_age_study_dashboard.html"
    df = pd.read_csv(INPUT)
    enriched, cols = add_age_score(df, score_map, source)
    enriched.to_csv(OUTPUT, index=False)

    targets = detect_targets(enriched)
    adp_col = cols.get("adp")
    validation = run_walk_forward(enriched, targets, adp_col)
    bucket = run_bucket_validation(enriched, targets)
    draft = run_draft_window_validation(enriched, targets, adp_col)
    combined = pd.concat([validation, bucket, draft], ignore_index=True, sort=False)
    combined.to_csv(VALIDATION_OUTPUT, index=False)
    classification, reason, best, best_draft = classify_signal(validation, draft)

    meta = {"dashboard": dashboard, "related_files": related, "used_files": used_files, "fallback_needed": fallback_needed}
    write_feature_report(meta, score_frames, cols, enriched)
    write_validation_report(validation, bucket, draft, targets, classification, reason, best, best_draft)

    sample_cols = [c for c in [cols.get("player"), cols.get("season"), cols.get("position"), cols.get("age"), cols.get("adp"), "age_curve_edge_score", "age_curve_edge_bucket", "age_curve_edge_note"] if c]
    print(f"rows loaded: {len(df)}")
    print(f"rows saved: {len(enriched)}")
    print(f"seasons covered: {sorted(pd.to_numeric(enriched[cols['season']], errors='coerce').dropna().astype(int).unique().tolist())[:5]} ... {sorted(pd.to_numeric(enriched[cols['season']], errors='coerce').dropna().astype(int).unique().tolist())[-5:]}")
    print(f"positions covered: {sorted(enriched[cols['position']].dropna().astype(str).unique().tolist())}")
    print(f"detected age column: {cols.get('age')}")
    print(f"detected position column: {cols.get('position')}")
    print(f"detected season column: {cols.get('season')}")
    print(f"detected ADP column: {cols.get('adp')}")
    print(f"detected positional ADP column: {cols.get('positional_adp')}")
    print(f"fantasy_age_study_dashboard.html found: {dashboard['dashboard_found']}")
    print(f"dashboard parsed successfully: {dashboard['dashboard_parsed_successfully']}")
    for pos, frame in score_frames.items():
        top = frame.sort_values("age_curve_edge_score", ascending=False).head(3)
        print(f"{pos} extracted age findings: " + ", ".join(f"age {int(r.Age)} score {r.age_curve_edge_score:.1f}" for _, r in top.iterrows()))
    print(f"related source files found: {len(related)}")
    print(f"related source files used: {used_files}")
    print(f"fallback curves needed: {fallback_needed}")
    print(f"valid age scores: {int(enriched['age_curve_edge_score'].notna().sum())}")
    print(f"missing age scores: {int(enriched['age_curve_edge_score'].isna().sum())}")
    print("first 20 scored rows:")
    print(enriched[sample_cols].head(20).to_string(index=False))
    print(f"validation targets available: {targets}")
    print("best validation result:")
    print(best.to_string() if best is not None else "none")
    print("best draft-window result:")
    print(best_draft.to_string() if best_draft is not None else "none")
    beat = bool(best is not None and best.get("improvement_over_adp_only", 0) > 0)
    print(f"ADP + age beat ADP-only: {'yes' if beat else 'no'}")
    print(f"final classification: {classification}")
    print(f"feature file created: {OUTPUT}")
    print(f"feature report created: {FEATURE_REPORT}")
    print(f"validation file created: {VALIDATION_OUTPUT}")
    print(f"validation report created: {VALIDATION_REPORT}")


if __name__ == "__main__":
    main()

