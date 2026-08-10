from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import re
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - scripts must still explain missing sklearn
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    roc_auc_score = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = VALIDATION_DIR / "predraft_validation_dataset.csv"

ADP_BUCKETS = [
    ("1-24", 1, 24),
    ("25-48", 25, 48),
    ("49-72", 49, 72),
    ("73-96", 73, 96),
    ("97-120", 97, 120),
    ("121-168", 121, 168),
    ("169+", 169, math.inf),
    ("Missing", None, None),
]

POSITIONAL_ADP_BUCKETS = [
    ("1-12", 1, 12),
    ("13-24", 13, 24),
    ("25-36", 25, 36),
    ("37-48", 37, 48),
    ("49+", 49, math.inf),
    ("Missing", None, None),
]


def clean_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace(".", "").replace("'", "").replace(" jr", "").replace(" sr", "")



def initial_last_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace(".", " ").replace("'", "")
    text = text.replace(" jr", "").replace(" sr", "")
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0][0]} {tokens[-1]}"

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def rank_within_position(df: pd.DataFrame, value_col: str, position_col: str = "position") -> pd.Series:
    return df.groupby(["season", position_col])[value_col].rank(ascending=False, method="first")


def label_bucket(value: object, buckets: list[tuple[str, float | None, float | None]]) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "Missing"
    for label, low, high in buckets:
        if low is None:
            continue
        if float(val) >= float(low) and float(val) <= float(high):
            return label
    return "Missing"


def top_decile_threshold_count(n: int) -> int:
    return max(1, int(math.ceil(n * 0.10))) if n else 0


def auc_or_nan(y_true: pd.Series, scores: pd.Series) -> float:
    if roc_auc_score is None:
        return float("nan")
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(scores, errors="coerce")
    mask = y.notna() & s.notna()
    if mask.sum() < 2 or y[mask].nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y[mask], s[mask]))
    except Exception:
        return float("nan")


def hit_rate(df: pd.DataFrame, target: str) -> float:
    if df.empty or target not in df.columns:
        return float("nan")
    y = pd.to_numeric(df[target], errors="coerce")
    if y.notna().sum() == 0:
        return float("nan")
    return float(y.mean())


def model_specs(feature_groups: dict[str, list[str]], include_random_forest: bool = True) -> list[dict[str, object]]:
    specs = []
    for group_name, features in feature_groups.items():
        if not features:
            continue
        specs.append({"model_name": f"{group_name}_logistic", "model_type": "Logistic Regression", "features": features, "kind": "logistic"})
        specs.append({"model_name": f"{group_name}_regularized_logistic", "model_type": "Regularized Logistic Regression", "features": features, "kind": "regularized_logistic"})
        if include_random_forest:
            specs.append({"model_name": f"{group_name}_random_forest", "model_type": "Random Forest", "features": features, "kind": "random_forest"})
    return specs


# Extra model kinds run ONLY for the comprehensive stacked feature group
# (adp_all_v1), not every group. "regularized_logistic" above uses sklearn's
# default L2 penalty, which shrinks coefficients but never actually drops
# one -- so a large stacked group gets no real feature selection, just
# dilution from whichever sparse/noisy features are in the mix. lasso_logistic
# (L1) genuinely zeroes out unhelpful features. hist_gradient_boosting
# handles missing values natively instead of median-imputing them, which
# matters here because several stacked features are >60-90% NaN. Scoped to
# one group rather than all of them because every evaluator run already
# takes 5+ minutes across ~12 groups x 3 kinds -- the question being asked
# is specifically about the full stack, not every individual block.
STACKED_WEIGHT_KINDS = ("lasso_logistic", "hist_gradient_boosting")


def stacked_model_specs(group_name: str, features: list[str]) -> list[dict[str, object]]:
    if not features:
        return []
    return [
        {"model_name": f"{group_name}_lasso_logistic", "model_type": "Lasso Logistic Regression", "features": features, "kind": "lasso_logistic"},
        {"model_name": f"{group_name}_hist_gradient_boosting", "model_type": "Histogram Gradient Boosting", "features": features, "kind": "hist_gradient_boosting"},
    ]


def build_estimator(kind: str):
    if kind in {"logistic", "regularized_logistic"}:
        if LogisticRegression is None or Pipeline is None or StandardScaler is None:
            return None
        c_value = 1.0 if kind == "logistic" else 0.35
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c_value, max_iter=1000, class_weight="balanced", solver="liblinear")),
        ])
    if kind == "lasso_logistic":
        if LogisticRegression is None or Pipeline is None or StandardScaler is None:
            return None
        # l1_ratio=1.0 (not penalty="l1") -- this sklearn version deprecated
        # the `penalty` kwarg in favor of `l1_ratio`; passing both raises a
        # FutureWarning and an inconsistent-values warning.
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(l1_ratio=1.0, C=0.35, max_iter=1000, class_weight="balanced", solver="liblinear")),
        ])
    if kind == "random_forest":
        if RandomForestClassifier is None:
            return None
        return RandomForestClassifier(n_estimators=250, min_samples_leaf=8, random_state=42, class_weight="balanced_subsample")
    if kind == "hist_gradient_boosting":
        if HistGradientBoostingClassifier is None:
            return None
        return HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=15, random_state=42)
    return None


def _extract_feature_weights(estimator, feature_names: list[str], kind: str) -> dict[str, float] | None:
    """
    Per-feature weight/importance for the stacked-model introspection CSV.

    Only lasso_logistic is supported: its L1 coefficients are a direct,
    cheap answer to "did this feature earn real weight in combination."
    hist_gradient_boosting has no equivalent built-in attribute (unlike
    RandomForestClassifier, sklearn's HistGradientBoostingClassifier does
    not expose feature_importances_) -- getting importances out of it would
    require sklearn.inspection.permutation_importance, which refits/scores
    repeatedly per season and wasn't worth the added runtime for a model
    that's already being tested purely for whether its lift beats ADP, not
    for interpretability.
    """
    if kind != "lasso_logistic":
        return None
    try:
        model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
        coefs = model.coef_[0]
        return {name: float(c) for name, c in zip(feature_names, coefs)}
    except Exception:
        return None


def append_stacked_weight_rows(weight_rows: list[dict[str, object]], position: str) -> None:
    """
    Read-filter-concat-write into one shared CSV across all 4 position
    evaluators, same pattern as build_workload_features_v1.py's
    _append_feature_inventory_rows() -- each position's rows replace only
    that position's prior rows, so re-running one evaluator doesn't lose
    the others' results.
    """
    if not weight_rows:
        return
    out_path = VALIDATION_DIR / "data" / "stacked_model_feature_weights.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(weight_rows)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["position"] != position]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(out_path, index=False)


def simple_score(df: pd.DataFrame, features: list[str]) -> pd.Series:
    if not features:
        return pd.Series(np.nan, index=df.index)
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    ranked = pd.DataFrame(index=df.index)
    for col in numeric.columns:
        values = numeric[col]
        if values.notna().sum() == 0:
            ranked[col] = 0.0
        else:
            ranked[col] = values.rank(pct=True).fillna(0.5)
    return ranked.mean(axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, features: list[str], kind: str, collect_weights: bool = False):
    """
    Returns (scores, status) normally, or (scores, status, weights) when
    collect_weights=True -- weights is a {feature: weight} dict for
    lasso_logistic, else None. Variable-arity return so every existing
    2-value-unpacking call site keeps working unchanged; only call sites
    that explicitly request weights need to unpack 3 values.
    """
    def _return(scores: pd.Series, status: str, weights: dict[str, float] | None = None):
        return (scores, status, weights) if collect_weights else (scores, status)

    available = [f for f in features if f in train.columns and train[f].notna().any() and test[f].notna().any()]
    if not available:
        return _return(pd.Series(np.nan, index=test.index), "skipped_no_features")
    train_y = pd.to_numeric(train[target], errors="coerce")
    if train_y.nunique(dropna=True) < 2:
        return _return(pd.Series(np.nan, index=test.index), "skipped_one_class_train")
    train_x = train[available].apply(pd.to_numeric, errors="coerce")
    test_x = test[available].apply(pd.to_numeric, errors="coerce")
    # hist_gradient_boosting handles NaN natively -- median-imputing ahead of
    # it would defeat the entire point of using it (see stacked_model_specs()
    # docstring), so it gets the raw frame instead of the filled one.
    if kind != "hist_gradient_boosting":
        medians = train_x.median(numeric_only=True).fillna(0.0)
        train_x = train_x.fillna(medians)
        test_x = test_x.fillna(medians)
    estimator = build_estimator(kind)
    if estimator is None:
        return _return(simple_score(test, available), "fallback_rank_score_no_sklearn")
    try:
        estimator.fit(train_x, train_y)
        if hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(test_x)[:, 1]
        else:
            scores = estimator.predict(test_x)
        weights = _extract_feature_weights(estimator, available, kind) if collect_weights else None
        return _return(pd.Series(scores, index=test.index), "fit", weights)
    except Exception as exc:
        return _return(simple_score(test, available), f"fallback_rank_score_error:{exc.__class__.__name__}")


def adp_baseline_scores(df: pd.DataFrame) -> pd.Series:
    if "overall_adp" in df.columns and df["overall_adp"].notna().any():
        return -pd.to_numeric(df["overall_adp"], errors="coerce")
    if "preseason_adp" in df.columns and df["preseason_adp"].notna().any():
        return -pd.to_numeric(df["preseason_adp"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def summarize_model(test: pd.DataFrame, target: str, scores: pd.Series, baseline_scores: pd.Series, test_season: int, model_name: str, model_type: str, feature_group: str, status: str) -> dict[str, object]:
    scored = test.copy()
    scored["model_score"] = scores
    scored["adp_baseline_score"] = baseline_scores
    valid_scores = scored["model_score"].notna()
    n = int(len(scored))
    top_n = top_decile_threshold_count(n)
    model_top = scored[valid_scores].sort_values("model_score", ascending=False).head(top_n) if top_n else scored.iloc[0:0]
    adp_top = scored[scored["adp_baseline_score"].notna()].sort_values("adp_baseline_score", ascending=False).head(top_n) if top_n else scored.iloc[0:0]
    model_hit = hit_rate(model_top, target)
    baseline_hit = hit_rate(adp_top, target)
    return {
        "test_season": test_season,
        "target": target,
        "model_name": model_name,
        "model_type": model_type,
        "feature_group": feature_group,
        "status": status,
        "sample_size": n,
        "positive_rate": hit_rate(scored, target),
        "baseline_hit_rate": baseline_hit,
        "model_hit_rate": model_hit,
        "lift_over_baseline": model_hit - baseline_hit if pd.notna(model_hit) and pd.notna(baseline_hit) else np.nan,
        "auc": auc_or_nan(scored[target], scored["model_score"]),
        "adp_auc": auc_or_nan(scored[target], scored["adp_baseline_score"]),
        "top_decile_hit_rate": model_hit,
        "adp_available": bool(scored["adp_baseline_score"].notna().any()),
        "beats_adp": bool(pd.notna(model_hit) and pd.notna(baseline_hit) and model_hit > baseline_hit),
    }


def top_pick_rows(test: pd.DataFrame, target: str, scores: pd.Series, baseline_scores: pd.Series, model_name: str, test_season: int, top_n: int = 20) -> pd.DataFrame:
    out = test.copy()
    out["model_score"] = scores
    out["adp_baseline_score"] = baseline_scores
    if out["model_score"].notna().sum() == 0:
        return pd.DataFrame()
    out = out.sort_values("model_score", ascending=False).head(top_n).copy()
    cols = [
        "season", "player_name", "player_id", "position", "team", "age", "preseason_adp", "positional_adp", "overall_adp",
        "estimated_draft_round", "model_score", "adp_baseline_score", target, "final_positional_finish", "final_fantasy_points",
        "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_targets", "prior_carries",
    ]
    out = ensure_columns(out, cols)
    out["test_season"] = test_season
    out["target"] = target
    out["model_name"] = model_name
    out["rank"] = range(1, len(out) + 1)
    return out[["test_season", "target", "model_name", "rank"] + cols]


def bucket_rows(test: pd.DataFrame, target: str, scores: pd.Series, test_season: int, model_name: str) -> list[dict[str, object]]:
    out = test.copy()
    out["model_score"] = scores
    out["overall_adp_bucket"] = out.get("overall_adp", pd.Series(np.nan, index=out.index)).apply(lambda x: label_bucket(x, ADP_BUCKETS))
    out["positional_adp_bucket"] = out.get("positional_adp", pd.Series(np.nan, index=out.index)).apply(lambda x: label_bucket(x, POSITIONAL_ADP_BUCKETS))
    rows = []
    for bucket_type, bucket_col in [("overall_adp", "overall_adp_bucket"), ("positional_adp", "positional_adp_bucket")]:
        for bucket, group in out.groupby(bucket_col, dropna=False):
            n = len(group)
            top_n = top_decile_threshold_count(n)
            model_top = group[group["model_score"].notna()].sort_values("model_score", ascending=False).head(top_n)
            rows.append({
                "test_season": test_season,
                "target": target,
                "model_name": model_name,
                "bucket_type": bucket_type,
                "bucket": bucket,
                "sample_size": int(n),
                "bucket_hit_rate": hit_rate(group, target),
                "model_top_decile_hit_rate": hit_rate(model_top, target),
            })
    return rows

