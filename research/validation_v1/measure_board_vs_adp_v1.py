"""
Step 9 Part A: how far does the SHIPPING board actually deviate from ADP?

Steps 4-5c established that ADP is very hard to beat and, more importantly,
that large deviations from it are actively harmful -- pure-model ranking
(lambda=1.0) cost -0.075 AUC at QB, and the measured optimum was a small
lambda ~= 0.10-0.25. That was all measured on the research harness.

Nothing has ever measured the score the app actually displays. Its weights
(DEFAULT_MASTER_COMPONENT_WEIGHTS in draftkit/draft_analysis.py) are
hand-set, ADP is never a sort key there, and 40% of the score is a
position-constant block offset. So the open question is simply: where does
the shipped board sit on that same lambda axis?

This script answers it descriptively -- no outcomes needed, no historical
projections needed (there are none; see Step 9 notes). It runs the REAL
build_recommendation_rankings_df() against the live 2026 pool and measures
displacement from ADP directly.

Running the app's scoring outside Streamlit: draft_analysis.py does
`import streamlit as st` and reads `st.session_state` in 6 places, always
via .get() with a default. Since those are attribute lookups on the module
at call time, replacing streamlit.session_state with a plain dict before
calling is sufficient -- the scoring math itself is pure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# data_access.get_csv_path() resolves RELATIVE paths ("data/processed/
# master_players.csv"), so it silently returns None -- and the board comes
# back empty -- unless the process CWD is the project root.
os.chdir(PROJECT_ROOT)

import streamlit as st  # noqa: E402

# Default league config, matching the app's own sidebar defaults. Stubbed
# BEFORE importing draft_analysis so nothing reads a missing key.
DEFAULT_SESSION = {
    "roster_settings": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 6},
    "my_team": [],
    "drafted_players": [],
    "current_pick_number": 1,
    "league_size": 12,
    "my_draft_slot": 1,
}


class _SessionStub(dict):
    """dict with attribute access, matching how st.session_state is used."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


st.session_state = _SessionStub(DEFAULT_SESSION)  # type: ignore[assignment]

from draftkit.draft_analysis import build_recommendation_rankings_df  # noqa: E402

LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
OUTPUT_PATH = Path(__file__).resolve().parent / "board_vs_adp_report.csv"
DISAGREEMENT_PATH = Path(__file__).resolve().parent / "board_vs_adp_disagreements.csv"


def _pct_rank_better_is_higher(series: pd.Series, lower_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return (-vals if lower_is_better else vals).rank(pct=True)


def _overlap(a: pd.Series, b: pd.Series, n: int) -> float:
    top_a = set(a.head(n))
    top_b = set(b.head(n))
    return len(top_a & top_b) / n if n else float("nan")


def analyze(board: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = board.copy()
    if "adp" not in df.columns or "final_score" not in df.columns:
        raise SystemExit(f"Expected 'adp' and 'final_score'; got {list(df.columns)[:25]}")

    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    # Only players the market actually prices can be compared to the market.
    df = df[df["adp"].notna() & df["final_score"].notna()].copy()

    df["adp_pct"] = _pct_rank_better_is_higher(df["adp"], lower_is_better=True)
    df["board_pct"] = _pct_rank_better_is_higher(df["final_score"], lower_is_better=False)
    df["adp_rank"] = df["adp"].rank(method="first")
    df["board_rank"] = (-df["final_score"]).rank(method="first")
    df["rank_shift"] = df["board_rank"] - df["adp_rank"]

    by_adp = df.sort_values("adp_rank")["player_name"].reset_index(drop=True)
    by_board = df.sort_values("board_rank")["player_name"].reset_index(drop=True)

    rows = [{
        "scope": "ALL",
        "n": len(df),
        "spearman_vs_adp": float(df["board_rank"].corr(df["adp_rank"], method="spearman")),
        "top25_overlap": _overlap(by_board, by_adp, 25),
        "top50_overlap": _overlap(by_board, by_adp, 50),
        "top100_overlap": _overlap(by_board, by_adp, 100),
        "mean_abs_rank_shift": float(df["rank_shift"].abs().mean()),
        "median_abs_rank_shift": float(df["rank_shift"].abs().median()),
        "max_abs_rank_shift": float(df["rank_shift"].abs().max()),
    }]

    for position, grp in df.groupby("position"):
        if len(grp) < 5:
            continue
        g_adp = grp.sort_values("adp_rank")["player_name"].reset_index(drop=True)
        g_board = grp.sort_values("board_rank")["player_name"].reset_index(drop=True)
        rows.append({
            "scope": str(position),
            "n": len(grp),
            "spearman_vs_adp": float(grp["board_rank"].corr(grp["adp_rank"], method="spearman")),
            "top25_overlap": _overlap(g_board, g_adp, min(25, len(grp))),
            "top50_overlap": _overlap(g_board, g_adp, min(50, len(grp))),
            "top100_overlap": float("nan"),
            "mean_abs_rank_shift": float(grp["rank_shift"].abs().mean()),
            "median_abs_rank_shift": float(grp["rank_shift"].abs().median()),
            "max_abs_rank_shift": float(grp["rank_shift"].abs().max()),
        })

    return pd.DataFrame(rows), df


def implied_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Locate the shipped board on the Step 5c lambda axis.

    Step 5c swept `final = (1-lam)*adp_pct + lam*model_pct`. Reconstruct that
    same blend on the current pool using projection as the model signal (the
    app's score is projection-dominated), then find which lam produces the
    ranking most similar to the board's actual ranking. That gives an
    apples-to-apples read of how far from consensus the app is operating,
    against a cost curve that has already been measured rather than guessed.
    """
    proj = pd.to_numeric(df.get("projection_points"), errors="coerce")
    if proj.notna().sum() < 20:
        return pd.DataFrame()
    model_pct = proj.rank(pct=True)

    rows = []
    for lam in LAMBDA_GRID:
        blended = (1.0 - float(lam)) * df["adp_pct"] + float(lam) * model_pct
        mask = blended.notna() & df["board_pct"].notna()
        rows.append({
            "lam": float(lam),
            "spearman_to_board": float(blended[mask].corr(df["board_pct"][mask], method="spearman")),
            "spearman_to_adp": float(blended[mask].corr(df["adp_pct"][mask], method="spearman")),
        })
    return pd.DataFrame(rows)


def main() -> None:
    board = build_recommendation_rankings_df()
    if board.empty:
        raise SystemExit("build_recommendation_rankings_df() returned empty -- check master_players.csv")
    print(f"Board rows scored: {len(board)}")

    summary, scored = analyze(board)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nReport written: {OUTPUT_PATH}")

    print("\n=== SHIPPED BOARD vs ADP ===")
    print(summary.round(3).to_string(index=False))

    print("\n=== Biggest disagreements (board much HIGHER than ADP = board loves them) ===")
    risers = scored.nsmallest(12, "rank_shift")[["player_name", "position", "adp", "adp_rank", "board_rank", "rank_shift", "final_score"]]
    print(risers.round(1).to_string(index=False))

    print("\n=== Biggest disagreements (board much LOWER than ADP = board fades them) ===")
    fallers = scored.nlargest(12, "rank_shift")[["player_name", "position", "adp", "adp_rank", "board_rank", "rank_shift", "final_score"]]
    print(fallers.round(1).to_string(index=False))

    lam_df = implied_lambda(scored)
    if not lam_df.empty:
        best = lam_df.loc[lam_df["spearman_to_board"].idxmax()]
        print("\n=== Implied position on the Step 5c lambda axis ===")
        print(lam_df.round(3).to_string(index=False))
        print(f"\nBoard most resembles a lambda = {best['lam']:.2f} blend "
              f"(spearman {best['spearman_to_board']:.3f} to the actual board).")
        print("Step 5c measured the OPTIMUM at lambda ~= 0.10-0.25, and lambda=1.0 "
              "at -0.075 AUC (QB) / -0.072 (RB).")


if __name__ == "__main__":
    main()
