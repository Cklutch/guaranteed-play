"""Real backfill data-coverage report (rookie_data_backfill_plan.pdf's Task A,
Step 2 -- built fresh, no such report existed anywhere in the repo before this).

Cross-references data/processed/backfill_candidates.csv (built by
find_backfill_candidates.py) against the real combine + season-stats files
already gathered this session, so the next sourcing pass knows exactly who
still needs real data instead of re-requesting what's already on hand.

Real, confirmed gap surfaced by this report: no rb_season_stats_2024.csv
exists anywhere in data/raw/ (only rb_combine_measurables_2024.csv) -- every
2024 RB candidate has combine measurables but zero real dominator-rating
source, unlike every other season/position combo in scope.

Usage:
    python -m draftkit.scripts.report_backfill_data_coverage
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.data_pipeline import normalize_player_name  # noqa: E402

CANDIDATES_CSV = REPO_ROOT / "data" / "processed" / "backfill_candidates.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"

# (season, position) -> (combine_csv, season_stats_csv). None marks a real,
# confirmed absence of that source -- not an oversight.
SOURCE_FILES = {
    (2024, "WR"): (RAW_DIR / "wr_combine_measurables_2024.csv", RAW_DIR / "wr_season_stats_2024.csv"),
    (2024, "RB"): (RAW_DIR / "rb_combine_measurables_2024.csv", None),
    (2025, "WR"): (RAW_DIR / "rb_wr_combine_measurables_2025.csv", RAW_DIR / "rb_wr_season_stats_2025.csv"),
    (2025, "RB"): (RAW_DIR / "rb_wr_combine_measurables_2025.csv", RAW_DIR / "rb_wr_season_stats_2025.csv"),
}


def _keys_from(csv_path: Path | None, name_col: str = "Player") -> set[str]:
    if csv_path is None or not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df[name_col].apply(normalize_player_name))


def build_coverage_report() -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATES_CSV)
    candidates["_key"] = candidates["player_name"].apply(normalize_player_name)

    rows = []
    for (season, position), (combine_csv, stats_csv) in SOURCE_FILES.items():
        combine_keys = _keys_from(combine_csv)
        stats_keys = _keys_from(stats_csv)
        subset = candidates[(candidates["season"] == season) & (candidates["position"] == position)]
        for _, row in subset.iterrows():
            rows.append({
                "season": season,
                "position": position,
                "player_name": row["player_name"],
                "has_combine": row["_key"] in combine_keys,
                "has_season_stats": row["_key"] in stats_keys,
            })

    return pd.DataFrame(rows)


def main() -> int:
    report = build_coverage_report()
    total = len(report)
    both = int((report["has_combine"] & report["has_season_stats"]).sum())
    combine_only = int((report["has_combine"] & ~report["has_season_stats"]).sum())
    neither = int((~report["has_combine"] & ~report["has_season_stats"]).sum())

    print(f"[report] {total} real backfill candidates")
    print(f"  already fully sourced (combine + season stats): {both}")
    print(f"  combine only, still need season stats:          {combine_only}")
    print(f"  neither -- still need full sourcing:             {neither}")
    print()
    print("By season/position:")
    print(
        report.groupby(["season", "position"])[["has_combine", "has_season_stats"]]
        .sum()
        .assign(total=report.groupby(["season", "position"]).size())
        .to_string()
    )
    print()
    still_needed = report[~(report["has_combine"] & report["has_season_stats"])]
    print(f"Still need sourcing ({len(still_needed)}):")
    print(still_needed[["season", "position", "player_name", "has_combine", "has_season_stats"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
