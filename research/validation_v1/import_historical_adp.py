from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from validation_utils import PROJECT_ROOT, VALIDATION_DIR, clean_name, initial_last_key

OUTPUT_COLUMNS = [
    "season",
    "player_name",
    "position",
    "overall_adp",
    "positional_adp",
    "preseason_projection",
    "projection_source",
    "adp_source",
    "half_ppr_adp",
    "ppr_adp",
    "standard_adp",
]

COLUMN_CANDIDATES = {
    "season": ["season", "year", "draft_year"],
    "player_name": ["player_name", "player", "name", "player name", "full_name"],
    "position": ["position", "pos", "fantasy_position"],
    "overall_adp": ["overall_adp", "adp", "average_draft_position", "average draft position", "avg_pick", "overall", "overall_rank"],
    "positional_adp": ["positional_adp", "pos_adp", "position_adp", "position_rank", "pos_rank", "rank_pos"],
    "preseason_projection": ["preseason_projection", "projection", "projection_points", "projected_points", "fpts", "points"],
    "projection_source": ["projection_source", "proj_source", "projection provider"],
    "adp_source": ["adp_source", "source", "provider", "site"],
    "half_ppr_adp": ["half_ppr_adp", "half_ppr", "half ppr adp", "0.5_ppr_adp"],
    "ppr_adp": ["ppr_adp", "ppr", "ppr adp"],
    "standard_adp": ["standard_adp", "std_adp", "standard", "standard adp"],
}

VALID_POSITIONS = {"WR", "RB", "QB", "TE"}


def _find_input_files(input_paths: list[str], input_dir: str | None) -> list[Path]:
    files: list[Path] = []
    for raw in input_paths:
        path = Path(raw)
        if path.exists() and path.is_file():
            files.append(path)
    if input_dir:
        root = Path(input_dir)
        if root.exists():
            for csv_path in sorted(root.glob("*.csv")):
                lower_name = csv_path.name.lower()
                if "manifest" in lower_name or lower_name.startswith("historical_adp_validation_"):
                    continue
                files.append(csv_path)
    seen = set()
    unique = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).lower().strip(): c for c in df.columns}
    rename = {}
    for target, candidates in COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate in lower:
                rename[lower[candidate]] = target
                break
    return df.rename(columns=rename)


def _read_one(path: Path, default_source: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    missing = [c for c in ["season", "player_name", "position"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns after normalization: {missing}")
    if "overall_adp" not in df.columns and "half_ppr_adp" not in df.columns and "ppr_adp" not in df.columns and "standard_adp" not in df.columns:
        raise ValueError(f"{path} needs overall_adp/adp or a scoring-specific ADP column.")
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OUTPUT_COLUMNS].copy()
    df["_input_file"] = str(path)
    if default_source:
        df["adp_source"] = df["adp_source"].fillna(default_source)
    df["adp_source"] = df["adp_source"].fillna(path.stem)
    return df


def _coerce_and_standardize(df: pd.DataFrame, scoring: str) -> pd.DataFrame:
    out = df.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["player_name"] = out["player_name"].astype(str).str.strip()
    out["position"] = out["position"].astype(str).str.upper().str.extract(r"(WR|RB|QB|TE)", expand=False)
    for col in ["overall_adp", "positional_adp", "preseason_projection", "half_ppr_adp", "ppr_adp", "standard_adp"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    score_col = {
        "half_ppr": "half_ppr_adp",
        "ppr": "ppr_adp",
        "standard": "standard_adp",
    }.get(scoring)
    if score_col and out["overall_adp"].isna().all() and out[score_col].notna().any():
        out["overall_adp"] = out[score_col]
    out["player_key"] = out["player_name"].apply(clean_name)
    out["initial_last_key"] = out["player_name"].apply(initial_last_key)
    return out


def _derive_positional_adp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = out["positional_adp"].isna() & out["overall_adp"].notna()
    if missing.any():
        derived = out[missing].groupby(["season", "position"])["overall_adp"].rank(method="first")
        out.loc[missing, "positional_adp"] = derived
    return out


def _dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_cols = ["season", "position", "player_key", "overall_adp", "positional_adp"]
    working = df.sort_values(sort_cols, na_position="last").copy()
    dup_mask = working.duplicated(["season", "position", "player_key"], keep=False)
    duplicates = working[dup_mask].copy()
    deduped = working.drop_duplicates(["season", "position", "player_key"], keep="first")
    return deduped, duplicates


def _load_outcome_keys() -> pd.DataFrame:
    path = PROJECT_ROOT / "research" / "validation_v1" / "predraft_validation_dataset.csv"
    if not path.exists():
        path = PROJECT_ROOT / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_ppr.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "position", "player_key"])
    df = pd.read_csv(path, usecols=lambda c: c in {"season", "position", "player_name"})
    df = df[df["position"].isin(["WR", "RB", "QB", "TE"])].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["position"] = df["position"].astype(str).str.upper()
    df["player_key"] = df["player_name"].apply(clean_name)
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)
    return df[["season", "position", "player_key", "initial_last_key"]].drop_duplicates()


def _validation_tables(df: pd.DataFrame, duplicates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    invalid_positions = df[~df["position"].isin(VALID_POSITIONS)].copy()
    missing_required = df[df[["season", "player_name", "position"]].isna().any(axis=1)].copy()
    missing_adp = df[df["overall_adp"].isna()].copy()
    suspicious_adp = df[df["overall_adp"].notna() & ((df["overall_adp"] <= 0) | (df["overall_adp"] > 400))].copy()
    suspicious_pos_adp = df[df["positional_adp"].notna() & ((df["positional_adp"] <= 0) | (df["positional_adp"] > 200))].copy()

    outcome_keys = _load_outcome_keys()
    unmatched = pd.DataFrame()
    if not outcome_keys.empty:
        keys = set(zip(outcome_keys["season"].astype(str), outcome_keys["position"].astype(str), outcome_keys["player_key"].astype(str)))
        tmp = df.copy()
        tmp["_key"] = list(zip(tmp["season"].astype(str), tmp["position"].astype(str), tmp["player_key"].astype(str)))
        unmatched = tmp[~tmp["_key"].isin(keys)].drop(columns=["_key"])

    coverage = df.groupby(["season", "position"], dropna=False).agg(
        rows=("player_name", "size"),
        rows_with_overall_adp=("overall_adp", lambda s: int(s.notna().sum())),
        rows_with_positional_adp=("positional_adp", lambda s: int(s.notna().sum())),
        median_overall_adp=("overall_adp", "median"),
        max_overall_adp=("overall_adp", "max"),
    ).reset_index()
    return {
        "coverage": coverage,
        "duplicates": duplicates,
        "invalid_positions": invalid_positions,
        "missing_required": missing_required,
        "missing_adp": missing_adp,
        "suspicious_adp": pd.concat([suspicious_adp, suspicious_pos_adp]).drop_duplicates(),
        "unmatched_name_examples": unmatched.head(200),
    }


def _write_report(tables: dict[str, pd.DataFrame], normalized: pd.DataFrame, output_path: Path, source_files: list[Path]) -> None:
    report_path = VALIDATION_DIR / "historical_adp_validation_report.md"
    seasons = sorted(normalized["season"].dropna().astype(int).unique().tolist())
    lines = [
        "# Historical ADP Validation Report",
        "",
        f"Output file: `{output_path}`",
        f"Source files: {', '.join(str(p) for p in source_files) if source_files else 'none'}",
        f"Rows imported: {len(normalized):,}",
        f"Seasons covered: {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)" if seasons else "Seasons covered: none",
        "",
        "## Validation Counts",
        "",
        f"Duplicate player-season-position rows found before dedupe: {len(tables['duplicates']):,}",
        f"Rows with missing overall ADP: {len(tables['missing_adp']):,}",
        f"Rows with invalid positions: {len(tables['invalid_positions']):,}",
        f"Rows with suspicious ADP values: {len(tables['suspicious_adp']):,}",
        f"Rows not matched to local outcome player names: {len(tables['unmatched_name_examples']):,} shown in sample file",
        "",
        "## Coverage By Season/Position",
        "",
        tables["coverage"].to_markdown(index=False) if not tables["coverage"].empty else "No coverage rows.",
        "",
        "## Decision",
        "",
    ]
    if len(normalized) == 0 or len(tables["missing_adp"]) == len(normalized) or len(tables["invalid_positions"]) > 0:
        lines.append("Not valid for model validation yet. Fix the issues above before relying on ADP comparisons.")
    else:
        lines.append("Structurally valid for import. Next step: run `build_predraft_dataset.py` and inspect merge coverage.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and validate historical preseason ADP CSVs for validation_v1.")
    parser.add_argument("--input", action="append", default=[], help="Input CSV path. May be passed more than once.")
    parser.add_argument("--input-dir", default=None, help="Directory containing source CSV files to combine.")
    parser.add_argument("--output", default=str(VALIDATION_DIR / "historical_adp.csv"))
    parser.add_argument("--scoring", choices=["half_ppr", "ppr", "standard"], default="half_ppr")
    parser.add_argument("--source", default=None, help="Default adp_source label if source files do not include one.")
    parser.add_argument("--derive-positional-adp", action="store_true", help="Derive positional_adp from overall_adp if missing.")
    args = parser.parse_args()

    files = _find_input_files(args.input, args.input_dir)
    if not files:
        print("No input ADP CSVs found. Use --input path.csv or --input-dir folder.")
        print("Template: research/validation_v1/historical_adp_TEMPLATE.csv")
        return

    frames = [_read_one(path, args.source) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = _coerce_and_standardize(combined, args.scoring)
    combined = combined[combined["position"].isin(VALID_POSITIONS)].copy()
    if args.derive_positional_adp:
        combined = _derive_positional_adp(combined)
    normalized, duplicates = _dedupe(combined)
    normalized = normalized[OUTPUT_COLUMNS + ["player_key", "initial_last_key", "_input_file"]].copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized[OUTPUT_COLUMNS].to_csv(output_path, index=False)

    tables = _validation_tables(normalized, duplicates)
    for name, table in tables.items():
        table.to_csv(VALIDATION_DIR / f"historical_adp_validation_{name}.csv", index=False)
    _write_report(tables, normalized, output_path, files)

    print(f"Historical ADP written: {output_path}")
    print(f"Rows imported: {len(normalized)}")
    print(f"Seasons covered: {sorted(normalized['season'].dropna().astype(int).unique().tolist())}")
    print(f"WR rows: {int(normalized['position'].eq('WR').sum())}")
    print(f"RB rows: {int(normalized['position'].eq('RB').sum())}")
    print(f"QB rows: {int(normalized['position'].eq('QB').sum())}")
    print(f"TE rows: {int(normalized['position'].eq('TE').sum())}")
    print(f"Duplicate rows before dedupe: {len(duplicates)}")
    print(f"Missing overall ADP rows: {len(tables['missing_adp'])}")
    print(f"Invalid position rows: {len(tables['invalid_positions'])}")
    print(f"Suspicious ADP rows: {len(tables['suspicious_adp'])}")
    print("Validation report: research/validation_v1/historical_adp_validation_report.md")


if __name__ == "__main__":
    main()



