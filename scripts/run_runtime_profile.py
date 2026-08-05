from __future__ import annotations

import json
import statistics
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeCache:
    def __call__(self, *decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def decorator(func):
            cache = {}

            def wrapper(*args, **kwargs):
                key = (
                    tuple(_cacheable(arg) for arg in args),
                    tuple(sorted((key, _cacheable(value)) for key, value in kwargs.items())),
                )
                if key not in cache:
                    cache[key] = func(*args, **kwargs)
                return cache[key]

            wrapper.clear = cache.clear
            return wrapper

        return decorator


def _cacheable(value):
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return ("df", tuple(value.columns), tuple(value.shape), hash(pd.util.hash_pandas_object(value, index=True).sum()))
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        return tuple(_cacheable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _cacheable(item)) for key, item in value.items()))
    try:
        hash(value)
        return value
    except Exception:
        return repr(value)


def _install_streamlit_shim():
    fake = types.ModuleType("streamlit")
    fake.session_state = {}
    fake.cache_data = _FakeCache()
    fake.cache_resource = _FakeCache()
    fake.set_page_config = lambda *args, **kwargs: None
    fake.sidebar = types.SimpleNamespace(checkbox=lambda *args, **kwargs: False)
    sys.modules["streamlit"] = fake
    return fake


def _event_stats(events):
    by_name = {}
    for event in events:
        by_name.setdefault(event["name"], []).append(event["elapsed_ms"])

    rows = []
    total_runtime = sum(sum(values) for values in by_name.values())
    for name, values in by_name.items():
        rows.append({
            "name": name,
            "runs": len(values),
            "average_ms": round(statistics.mean(values), 2),
            "max_ms": round(max(values), 2),
            "total_ms": round(sum(values), 2),
            "runtime_share_pct": round((sum(values) / total_runtime * 100.0) if total_runtime else 0.0, 2),
        })
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)


def main():
    _install_streamlit_shim()

    import pandas as pd

    from draftkit.performance_profiler import PerformanceProfiler
    from draftkit.data_access import get_available_players_df, load_players_df
    from draftkit.draft_analysis import build_recommendation_rankings_df
    from draftkit.conviction import build_conviction_report
    from draftkit.draft_simulation import calculate_availability_probability
    from draftkit.draft_simulator import build_simulation_report
    from draftkit.championship_equity_v2 import build_championship_equity_v2_df
    from draftkit.recommendation_consensus import build_consensus_recommendations
    from draftkit.construction_pressure import calculate_construction_pressure
    from draftkit.draft_strategy import build_draft_strategy
    from draftkit.draft_lab import run_simulation_batch

    import streamlit as st

    st.session_state.update({
        "drafted_players": [],
        "my_team": [],
        "player_queue": [],
        "draft_log": [],
        "league_size": 12,
        "my_draft_slot": 1,
        "current_pick_number": 1,
        "roster_settings": {
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "FLEX": 1,
            "DST": 1,
            "K": 1,
            "BENCH": 6,
        },
        "player_position_map": {},
    })

    profiler = PerformanceProfiler()
    snapshots = {}

    for pass_name in ["cold", "warm_1", "warm_2"]:
        with profiler.track("data_loading", pass_name=pass_name):
            raw_df = load_players_df()

        with profiler.track("dataframe_filtering", pass_name=pass_name):
            available_df = get_available_players_df()

        with profiler.track("recommendation_generation", pass_name=pass_name):
            recommendation_df = build_recommendation_rankings_df()

        with profiler.track("roster_construction_evaluation", pass_name=pass_name):
            construction_report = calculate_construction_pressure()

        with profiler.track("conviction_report", pass_name=pass_name):
            conviction_df = build_conviction_report(recommendation_df, limit=8)

        with profiler.track("availability_prediction", pass_name=pass_name, simulations=35):
            availability_df = calculate_availability_probability(num_simulations=35, seed=17)

        with profiler.track("championship_equity_v2", pass_name=pass_name, candidates=5, simulations=35):
            championship_df = build_championship_equity_v2_df(
                players_df=raw_df,
                candidates_df=recommendation_df.head(5),
                current_roster=[],
                draft_round=1,
                projected_roster_construction=st.session_state["roster_settings"],
                league_size=12,
                num_simulations=35,
                seed=53,
                limit=5,
            )

        with profiler.track("consensus_generation", pass_name=pass_name, candidates=12):
            consensus_report = build_consensus_recommendations(
                recommendations_df=recommendation_df,
                availability_df=availability_df,
                future_impact_df=pd.DataFrame(),
                championship_equity_df=championship_df,
                limit=12,
            )

        with profiler.track("future_impact_simulation", pass_name=pass_name, candidates=5, simulations=50):
            future_impact_df = build_simulation_report(
                recommendations_df=recommendation_df.head(5),
                limit=5,
                num_simulations=50,
                seed=29,
            )

        with profiler.track("draft_strategy", pass_name=pass_name):
            strategy_report = build_draft_strategy(recommendation_df)

        with profiler.track("rendering_preparation", pass_name=pass_name):
            summary_rows = recommendation_df.head(8).to_dict("records")
            board_rows = available_df.head(25).to_dict("records")
            consensus_rows = consensus_report.get("consensus_scores", [])

        snapshots[pass_name] = {
            "raw_rows": int(len(raw_df)),
            "available_rows": int(len(available_df)),
            "recommendation_rows": int(len(recommendation_df)),
            "conviction_rows": int(len(conviction_df)),
            "availability_rows": int(len(availability_df)),
            "championship_rows": int(len(championship_df)),
            "future_impact_rows": int(len(future_impact_df)),
            "construction_keys": sorted(construction_report.keys()),
            "strategy_keys": sorted(strategy_report.keys()),
            "summary_rows": len(summary_rows),
            "board_rows": len(board_rows),
            "consensus_rows": len(consensus_rows),
        }

    with profiler.track("draft_lab_batch_10", simulations=10):
        lab_result = run_simulation_batch(
            players_df=load_players_df(),
            draft_slot=1,
            league_size=12,
            strategy="Balanced",
            batch_size=10,
            seed=101,
        )

    summary = profiler.summary()
    payload = {
        "events": summary["events"],
        "stats": _event_stats(summary["events"]),
        "snapshots": snapshots,
        "draft_lab": {
            "simulations_run": lab_result.get("simulations_run"),
            "average_draft_grade": lab_result.get("average_draft_grade"),
            "average_championship_equity": lab_result.get("average_championship_equity"),
        },
    }

    output_path = Path("data/processed/runtime_profile_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["stats"], indent=2))


if __name__ == "__main__":
    main()
