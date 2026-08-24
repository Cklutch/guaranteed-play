"""Compare sportsbook-implied half-PPR fantasy points to the ADP-correlated projection.

Research/analytics use only -- not a betting tool. Sources sportsbook-derived
season-long projections from data/raw/winwithodds_season_projections.csv (see
draftkit/data_sources/winwithodds_source.py), which recomputes half-PPR points
(0.5 pts/reception) from WinWithOdds' raw per-category stats rather than
trusting their own Projections column (scored with an unknown/custom
setting). Compares against data/processed/master_players.csv's
`projection_points` column, which is the FantasyPros projection already tied to
`adp`/`adp_rank` and is itself already half-PPR scored.

This intentionally does NOT go through draftkit.sportsbook_provider's
fallback chain (which defaults to a mock provider) or
draftkit.projection_enrichment.merge_sportsbook_projections (which diffs
full-PPR sportsbook points against this same half-PPR projection_points
column) -- both would produce a mismatched or mocked comparison.

Note: draftkit/scripts/pull_sportsbook_props.py (live single-game props via
The Odds API) is a separate, independent data source -- still empty until
closer to kickoff. This script does not depend on it.

Output is limited to the top TOP_N players by adp_rank (the draftable pool) --
outside that range, projection_points is mostly 0/unranked in FantasyPros'
data and swamps the comparison with noise.

`adp_position_rank` and `sportsbook_position_rank` are positional order
labels (e.g. "QB7" = the 7th-highest QB by that ranking), matching how
fantasy players actually talk about rankings -- ADP's positional order vs.
the sportsbook-implied positional order. `position_rank_gap` is the signed
numeric difference (adp_position_rank_num - sportsbook_position_rank_num),
mirroring draftkit/market_disagreement.py's calculate_adp_gap sign
convention: positive means the sportsbook ranks the player better within
their position (lower number) than ADP does -- i.e. potential hidden value.
Negative means the opposite. Ranks are computed across the full WinWithOdds
pool (not just the top-N slice) so they reflect true positional standing.

Usage:
    python -m draftkit.scripts.compare_sportsbook_vs_adp
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from draftkit.data_access import load_players_df
from draftkit.data_sources.winwithodds_source import load_winwithodds_projections
from draftkit.projection_enrichment import (
    calculate_projection_disagreement,
    normalize_player_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "sportsbook_vs_adp_comparison.csv"

TOP_N = 300

OUTPUT_COLUMNS = [
    "player_name",
    "position",
    "team",
    "adp",
    "adp_rank",
    "projection_points",
    "sportsbook_half_ppr_points",
    "adp_position_rank",
    "sportsbook_position_rank",
    "position_rank_gap",
    "projection_gap",
    "projection_gap_pct",
    "categories_available",
]


def build_comparison_df() -> pd.DataFrame:
    sportsbook_df = load_winwithodds_projections()
    master_df = load_players_df()

    if master_df is None or master_df.empty or "player_name" not in master_df.columns:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    master = master_df.copy()
    master["normalized_player_name"] = master["player_name"].apply(normalize_player_name)

    keep_master_cols = [
        col for col in ("player_name", "position", "team", "adp", "adp_rank", "projection_points")
        if col in master.columns
    ]
    master = master[keep_master_cols + ["normalized_player_name"]]

    if sportsbook_df.empty:
        out = master.drop(columns=["normalized_player_name"]).copy()
        out["sportsbook_half_ppr_points"] = pd.NA
        out["adp_position_rank"] = pd.NA
        out["sportsbook_position_rank"] = pd.NA
        out["position_rank_gap"] = pd.NA
        out["projection_gap"] = pd.NA
        out["projection_gap_pct"] = pd.NA
        out["categories_available"] = 0
        out.attrs["sportsbook_row_count"] = 0
        out.attrs["matched_players"] = 0
        out = out[out["adp_rank"].notna() & (out["adp_rank"] <= TOP_N)] if "adp_rank" in out.columns else out
        return out[OUTPUT_COLUMNS]

    sportsbook = sportsbook_df.copy()
    sportsbook["normalized_player_name"] = sportsbook["player_name"].apply(normalize_player_name)
    sportsbook = sportsbook.rename(columns={"player_name": "sportsbook_player_name"})

    merged = master.merge(
        sportsbook[["normalized_player_name", "sportsbook_half_ppr_points", "categories_available"]],
        on="normalized_player_name",
        how="left",
    )

    # Rank within position, across the full matched pool (not the top-N slice below),
    # so labels like "QB7" reflect true positional standing, not a truncated view.
    adp_position_rank_num = merged.groupby("position")["adp"].rank(method="first", ascending=True)
    sportsbook_position_rank_num = merged.groupby("position")["sportsbook_half_ppr_points"].rank(
        method="first", ascending=False
    )
    merged["position_rank_gap"] = adp_position_rank_num - sportsbook_position_rank_num
    merged["adp_position_rank"] = merged["position"] + adp_position_rank_num.astype("Int64").astype(str)
    merged["sportsbook_position_rank"] = merged["position"] + sportsbook_position_rank_num.astype("Int64").astype(str)
    merged.loc[adp_position_rank_num.isna(), "adp_position_rank"] = pd.NA
    merged.loc[sportsbook_position_rank_num.isna(), "sportsbook_position_rank"] = pd.NA

    disagreement = merged.apply(
        lambda row: calculate_projection_disagreement(
            row.get("sportsbook_half_ppr_points"),
            row.get("projection_points"),
        ),
        axis=1,
    )
    merged["projection_gap"] = disagreement.apply(lambda item: item["projection_gap"])
    merged["projection_gap_pct"] = disagreement.apply(lambda item: item["projection_gap_pct"])
    merged["categories_available"] = merged["categories_available"].fillna(0).astype(int)

    merged.attrs["sportsbook_row_count"] = int(len(sportsbook_df))
    merged.attrs["matched_players"] = int(merged["sportsbook_half_ppr_points"].notna().sum())

    merged = merged[merged["adp_rank"].notna() & (merged["adp_rank"] <= TOP_N)]

    return merged.drop(columns=["normalized_player_name"])[OUTPUT_COLUMNS]


def validate_comparison(comparison_df: pd.DataFrame) -> dict:
    matched = comparison_df.attrs.get(
        "matched_players",
        int(comparison_df["sportsbook_half_ppr_points"].notna().sum()) if not comparison_df.empty else 0,
    )
    sportsbook_row_count = comparison_df.attrs.get("sportsbook_row_count", 0)
    messages = []
    if sportsbook_row_count == 0:
        messages.append(
            "No WinWithOdds projection data found. Confirm "
            "data/raw/winwithodds_season_projections.csv exists, then re-run this script."
        )
    return {
        "is_valid": matched > 0,
        "player_count": int(len(comparison_df)),
        "sportsbook_row_count": int(sportsbook_row_count),
        "matched_players": int(matched),
        "messages": messages,
    }


def main() -> int:
    comparison_df = build_comparison_df()
    validation = validate_comparison(comparison_df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[write] {OUTPUT_CSV}: {len(comparison_df)} row(s)")
    print(f"validation: {validation}")

    if not validation["is_valid"]:
        for message in validation["messages"]:
            print(f"[note] {message}")
        return 0

    matched_df = comparison_df.dropna(subset=["projection_gap"]).copy()
    display_cols = ["player_name", "position", "adp_position_rank", "sportsbook_position_rank",
                     "position_rank_gap", "projection_points", "sportsbook_half_ppr_points",
                     "projection_gap", "projection_gap_pct"]

    print(f"\nTop 15 (of top {TOP_N} by ADP): sportsbook higher than ADP-implied projection")
    higher = matched_df.sort_values("projection_gap", ascending=False).head(15)
    print(higher[display_cols].to_string(index=False))

    print(f"\nTop 15 (of top {TOP_N} by ADP): sportsbook lower than ADP-implied projection")
    lower = matched_df.sort_values("projection_gap", ascending=True).head(15)
    print(lower[display_cols].to_string(index=False))

    print(f"\nTop 15 (of top {TOP_N} by ADP): biggest positive position_rank_gap "
          f"(sportsbook ranks better within position than ADP)")
    rank_higher = matched_df.dropna(subset=["position_rank_gap"]).sort_values(
        "position_rank_gap", ascending=False
    ).head(15)
    print(rank_higher[display_cols].to_string(index=False))

    print(f"\nTop 15 (of top {TOP_N} by ADP): biggest negative position_rank_gap "
          f"(sportsbook ranks worse within position than ADP)")
    rank_lower = matched_df.dropna(subset=["position_rank_gap"]).sort_values(
        "position_rank_gap", ascending=True
    ).head(15)
    print(rank_lower[display_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
