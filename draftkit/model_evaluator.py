from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _name_key
from draftkit.common import safe_float as _safe_float


FEATURE_FIELDS = [
    "projection_score",
    "value_score",
    "risk_score",
    "roster_fit_score",
    "championship_equity",
    "future_impact_score",
]
STRATEGIES_TO_COMPARE = [
    "Hero RB",
    "Zero RB",
    "Balanced",
    "RB Heavy",
    "WR Heavy",
    "Early QB",
]
POSITIONS = ["QB", "RB", "WR", "TE"]


def _drafts_from_input(results: Any) -> List[Dict[str, Any]]:
    if results is None:
        return []
    if isinstance(results, dict):
        if "drafts" in results and isinstance(results["drafts"], list):
            return [draft for draft in results["drafts"] if isinstance(draft, dict)]
        if any(key in results for key in ["draft_history", "recommendation_history", "draft_grade"]):
            return [results]
    if isinstance(results, list):
        return [draft for draft in results if isinstance(draft, dict)]
    return []


def _mean(values: Iterable[Any], default: float = 0.0) -> float:
    numeric = [_safe_float(value, None) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return default
    return sum(numeric) / len(numeric)


def _pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return 0.0

    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _calibration_error(predicted: List[float], actual: List[float]) -> float:
    pairs = [
        (_safe_float(pred, None), _safe_float(outcome, None))
        for pred, outcome in zip(predicted, actual)
    ]
    pairs = [(pred, outcome) for pred, outcome in pairs if pred is not None and outcome is not None]
    if not pairs:
        return 0.0
    return round(sum(abs(pred - outcome) for pred, outcome in pairs) / len(pairs), 2)


def _ranking_accuracy(predicted: List[float], actual: List[float]) -> float:
    pairs = [
        (_safe_float(pred, None), _safe_float(outcome, None))
        for pred, outcome in zip(predicted, actual)
    ]
    pairs = [(pred, outcome) for pred, outcome in pairs if pred is not None and outcome is not None]
    if len(pairs) < 2:
        return 0.0

    correct = 0
    total = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            pred_delta = pairs[i][0] - pairs[j][0]
            actual_delta = pairs[i][1] - pairs[j][1]
            if pred_delta == 0 or actual_delta == 0:
                continue
            total += 1
            if (pred_delta > 0 and actual_delta > 0) or (pred_delta < 0 and actual_delta < 0):
                correct += 1
    if total <= 0:
        return 0.0
    return round((correct / total) * 100.0, 2)


def _position_counts(roster: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {position: 0 for position in POSITIONS}
    for player in roster or []:
        position = str(player.get("position", "")).upper()
        if position in counts:
            counts[position] += 1
    return counts


def _selected_pick_lookup(draft: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        _name_key(record.get("player_name")): record
        for record in draft.get("draft_history", [])
        if record.get("manager") == "You"
    }


def extract_recommendation_records(results: Any) -> pd.DataFrame:
    rows = []
    for draft_index, draft in enumerate(_drafts_from_input(results), start=1):
        grade = draft.get("draft_grade", {}) or {}
        selected_lookup = _selected_pick_lookup(draft)
        recommendation_history = draft.get("recommendation_history", []) or []
        for index, record in enumerate(recommendation_history, start=1):
            selected = record.get("selected_player")
            selected_pick = selected_lookup.get(_name_key(selected), {})
            rows.append({
                "draft_index": draft_index,
                "pick_number": record.get("pick_number"),
                "round": record.get("round"),
                "strategy": draft.get("strategy"),
                "draft_slot": draft.get("draft_slot"),
                "league_size": draft.get("league_size"),
                "player_selected": selected,
                "recommended_player": record.get("recommended_player"),
                "recommendation_rank": record.get("recommendation_rank", index),
                "consensus_score": _safe_float(record.get("consensus_score"), None),
                "championship_equity": _safe_float(record.get("championship_equity"), None),
                "confidence_score": _safe_float(
                    record.get("confidence_score", record.get("recommendation_confidence")),
                    None,
                ),
                "recommendation_followed": bool(record.get("recommendation_followed", False)),
                "final_draft_grade": _safe_float(grade.get("grade_score"), None),
                "final_championship_equity": _safe_float(grade.get("championship_equity"), None),
                "actual_outcome": _safe_float(grade.get("grade_score"), None),
                "selected_position": selected_pick.get("position"),
                "selected_adp": _safe_float(selected_pick.get("adp"), None),
                "projection_score": _safe_float(selected_pick.get("projection_score"), None),
                "value_score": _safe_float(selected_pick.get("value_score"), None),
                "risk_score": _safe_float(selected_pick.get("risk_score"), None),
                "roster_fit_score": _safe_float(grade.get("positional_balance"), None),
                "future_impact_score": _safe_float(record.get("future_impact_score"), None),
            })
    return pd.DataFrame(rows)


def _draft_grade_df(results: Any) -> pd.DataFrame:
    rows = []
    for draft_index, draft in enumerate(_drafts_from_input(results), start=1):
        grade = draft.get("draft_grade", {}) or {}
        rows.append({
            "draft_index": draft_index,
            "strategy": draft.get("strategy"),
            "draft_slot": draft.get("draft_slot"),
            "league_size": draft.get("league_size"),
            "grade_score": _safe_float(grade.get("grade_score"), 0.0),
            "championship_equity": _safe_float(grade.get("championship_equity"), 0.0),
            "risk_profile": _safe_float(grade.get("risk_profile"), 0.0),
            "value_gained": _safe_float(grade.get("value_gained"), 0.0),
            "roster_strength": _safe_float(grade.get("roster_strength"), 0.0),
            "positional_balance": _safe_float(grade.get("positional_balance"), 0.0),
            "portfolio_classification": grade.get("portfolio_classification"),
        })
    return pd.DataFrame(rows)


def build_feature_importance(results: Any) -> Dict[str, float]:
    records = extract_recommendation_records(results)
    if records.empty:
        return {field: round(1.0 / len(FEATURE_FIELDS), 4) for field in FEATURE_FIELDS}

    target = records["final_draft_grade"].fillna(records["actual_outcome"]).tolist()
    raw_importance = {}
    fallback_map = {
        "projection_score": "roster_strength",
        "value_score": None,
        "risk_score": None,
        "roster_fit_score": None,
        "championship_equity": "final_championship_equity",
        "future_impact_score": "consensus_score",
    }
    draft_grades = _draft_grade_df(results)

    for field in FEATURE_FIELDS:
        if field in records.columns and records[field].notna().any():
            values = records[field].tolist()
            importance = abs(_pearson(values, target))
        else:
            fallback = fallback_map.get(field)
            if fallback and not draft_grades.empty and fallback in draft_grades:
                importance = abs(_pearson(draft_grades[fallback].tolist(), draft_grades["grade_score"].tolist()))
            else:
                importance = 0.0
        raw_importance[field] = importance

    total = sum(raw_importance.values())
    if total <= 0:
        return {field: round(1.0 / len(FEATURE_FIELDS), 4) for field in FEATURE_FIELDS}
    return {
        field: round(value / total, 4)
        for field, value in raw_importance.items()
    }


def build_calibration_analysis(results: Any) -> Dict[str, Dict[str, float]]:
    records = extract_recommendation_records(results)
    if records.empty:
        empty = {"correlation": 0.0, "calibration_error": 0.0, "ranking_accuracy": 0.0}
        return {
            "consensus_vs_final_grade": empty.copy(),
            "confidence_vs_actual_outcome": empty.copy(),
            "championship_equity_vs_simulated_wins": empty.copy(),
        }

    final_grade = records["final_draft_grade"].fillna(0.0).tolist()
    simulated_equity = records["final_championship_equity"].fillna(0.0).tolist()
    consensus = records["consensus_score"].fillna(50.0).tolist()
    confidence = records["confidence_score"].fillna(50.0).tolist()
    championship_equity = records["championship_equity"].fillna(50.0).tolist()

    return {
        "consensus_vs_final_grade": {
            "correlation": _pearson(consensus, final_grade),
            "calibration_error": _calibration_error(consensus, final_grade),
            "ranking_accuracy": _ranking_accuracy(consensus, final_grade),
        },
        "confidence_vs_actual_outcome": {
            "correlation": _pearson(confidence, final_grade),
            "calibration_error": _calibration_error(confidence, final_grade),
            "ranking_accuracy": _ranking_accuracy(confidence, final_grade),
        },
        "championship_equity_vs_simulated_wins": {
            "correlation": _pearson(championship_equity, simulated_equity),
            "calibration_error": _calibration_error(championship_equity, simulated_equity),
            "ranking_accuracy": _ranking_accuracy(championship_equity, simulated_equity),
        },
    }


def detect_positional_bias(results: Any) -> Dict[str, Any]:
    drafts = _drafts_from_input(results)
    if not drafts:
        return {
            "findings": [],
            "position_rates": {},
            "bias_scores": {},
        }

    total_counts = Counter()
    early_counts = Counter()
    total_picks = 0
    early_picks = 0
    qb_reaches = 0
    te_early = 0

    for draft in drafts:
        for record in draft.get("draft_history", []):
            if record.get("manager") != "You":
                continue
            position = str(record.get("position", "")).upper()
            if position not in POSITIONS:
                continue
            total_counts[position] += 1
            total_picks += 1
            round_number = _safe_float(record.get("round"), 99.0)
            if round_number <= 5:
                early_counts[position] += 1
                early_picks += 1
            adp = _safe_float(record.get("adp"), record.get("pick_number"))
            pick_number = _safe_float(record.get("pick_number"), adp)
            if position == "QB" and pick_number - adp >= 12:
                qb_reaches += 1
            if position == "TE" and round_number <= 4:
                te_early += 1

    position_rates = {
        position: round(total_counts[position] / max(total_picks, 1) * 100.0, 2)
        for position in POSITIONS
    }
    early_rates = {
        position: round(early_counts[position] / max(early_picks, 1) * 100.0, 2)
        for position in POSITIONS
    }
    findings = []
    if position_rates["RB"] >= 42:
        findings.append("RB overweighting")
    if position_rates["WR"] >= 42:
        findings.append("WR overweighting")
    if qb_reaches / max(total_picks, 1) >= 0.08:
        findings.append("QB reach behavior")
    if te_early / max(len(drafts), 1) >= 0.35:
        findings.append("TE scarcity bias")

    return {
        "findings": findings or ["No major positional bias detected"],
        "position_rates": position_rates,
        "early_position_rates": early_rates,
        "bias_scores": {
            "RB overweighting": _clip_score((position_rates["RB"] - 30.0) * 3.0, default=0.0),
            "WR overweighting": _clip_score((position_rates["WR"] - 30.0) * 3.0, default=0.0),
            "QB reach behavior": round(qb_reaches / max(total_picks, 1) * 100.0, 2),
            "TE scarcity bias": round(te_early / max(len(drafts), 1) * 100.0, 2),
        },
    }


def compare_strategies(results: Any) -> pd.DataFrame:
    grades = _draft_grade_df(results)
    if grades.empty:
        return pd.DataFrame(columns=["strategy", "average_grade", "average_equity", "average_risk", "draft_count"])

    grouped = []
    for strategy in STRATEGIES_TO_COMPARE:
        subset = grades[grades["strategy"].astype(str) == strategy]
        if subset.empty:
            continue
        grouped.append({
            "strategy": strategy,
            "average_grade": round(float(subset["grade_score"].mean()), 2),
            "average_equity": round(float(subset["championship_equity"].mean()), 2),
            "average_risk": round(float(100.0 - subset["risk_profile"].mean()), 2),
            "draft_count": int(len(subset)),
        })

    return pd.DataFrame(grouped).sort_values(
        ["average_grade", "average_equity"],
        ascending=False,
    ).reset_index(drop=True) if grouped else pd.DataFrame(columns=["strategy", "average_grade", "average_equity", "average_risk", "draft_count"])


def analyze_draft_slots(results: Any) -> pd.DataFrame:
    grades = _draft_grade_df(results)
    if grades.empty:
        return pd.DataFrame(columns=["draft_slot", "average_equity", "average_grade", "average_value_captured", "draft_count"])

    rows = []
    for slot, subset in grades.groupby("draft_slot"):
        if pd.isna(slot):
            continue
        rows.append({
            "draft_slot": int(_safe_float(slot, 0.0)),
            "average_equity": round(float(subset["championship_equity"].mean()), 2),
            "average_grade": round(float(subset["grade_score"].mean()), 2),
            "average_value_captured": round(float(subset["value_gained"].mean()), 2),
            "draft_count": int(len(subset)),
        })
    if not rows:
        return pd.DataFrame(columns=["draft_slot", "average_equity", "average_grade", "average_value_captured", "draft_count"])
    return pd.DataFrame(rows).sort_values(
        ["average_grade", "average_equity"],
        ascending=False,
    ).reset_index(drop=True)


def build_failure_report(results: Any) -> pd.DataFrame:
    failure_rows = []
    player_counts = defaultdict(lambda: {
        "player": None,
        "failure_count": 0,
        "high_confidence_misses": 0,
        "low_confidence_misses": 0,
        "average_consensus_score": [],
        "average_final_grade": [],
    })

    for draft_index, draft in enumerate(_drafts_from_input(results), start=1):
        grade = draft.get("draft_grade", {}) or {}
        final_grade = _safe_float(grade.get("grade_score"), 0.0)
        for record in draft.get("recommendation_history", []) or []:
            followed = bool(record.get("recommendation_followed", False))
            confidence = _safe_float(
                record.get("confidence_score", record.get("recommendation_confidence")),
                50.0,
            )
            consensus = _safe_float(record.get("consensus_score"), 50.0)
            recommended = record.get("recommended_player")
            selected = record.get("selected_player")
            miss = not followed or final_grade < 65.0
            if not miss:
                continue

            key = _name_key(recommended or selected)
            item = player_counts[key]
            item["player"] = recommended or selected
            item["failure_count"] += 1
            item["average_consensus_score"].append(consensus)
            item["average_final_grade"].append(final_grade)
            if confidence >= 75 and not followed:
                item["high_confidence_misses"] += 1
            if confidence < 50:
                item["low_confidence_misses"] += 1

        for failure in draft.get("recommendation_failures", []) or []:
            failure_rows.append({
                "draft_index": draft_index,
                "type": failure.get("type"),
                "player": failure.get("player"),
                "severity": failure.get("severity"),
                "pick_number": failure.get("pick_number"),
            })

    rows = []
    for item in player_counts.values():
        rows.append({
            "player": item["player"],
            "failure_count": item["failure_count"],
            "high_confidence_misses": item["high_confidence_misses"],
            "low_confidence_misses": item["low_confidence_misses"],
            "average_consensus_score": round(_mean(item["average_consensus_score"]), 2),
            "average_final_grade": round(_mean(item["average_final_grade"]), 2),
        })

    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame(columns=[
            "player",
            "failure_count",
            "high_confidence_misses",
            "low_confidence_misses",
            "average_consensus_score",
            "average_final_grade",
        ])

    return report.sort_values(
        ["failure_count", "high_confidence_misses", "average_consensus_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def evaluate_draft_lab_results(results: Any) -> Dict[str, Any]:
    records_df = extract_recommendation_records(results)
    calibration = build_calibration_analysis(results)
    feature_importance = build_feature_importance(results)
    positional_bias = detect_positional_bias(results)
    strategy_rankings = compare_strategies(results)
    draft_slot_rankings = analyze_draft_slots(results)
    failure_report = build_failure_report(results)
    grades = _draft_grade_df(results)

    return {
        "sample_size": {
            "drafts": int(len(_drafts_from_input(results))),
            "recommendations": int(len(records_df)),
        },
        "recommendation_records": records_df,
        "feature_importance": feature_importance,
        "calibration": calibration,
        "positional_bias": positional_bias,
        "strategy_rankings": strategy_rankings,
        "draft_slot_rankings": draft_slot_rankings,
        "failure_report": failure_report,
        "summary": {
            "average_grade": round(float(grades["grade_score"].mean()), 2) if not grades.empty else 0.0,
            "average_equity": round(float(grades["championship_equity"].mean()), 2) if not grades.empty else 0.0,
            "average_value_captured": round(float(grades["value_gained"].mean()), 2) if not grades.empty else 0.0,
        },
    }
