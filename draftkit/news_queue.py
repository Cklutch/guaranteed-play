"""Light resolve queue for the news-override pipeline (Lane B / human
review -- see research/news_override_policy.md). The scheduled
nfl-camp-news-watch sweep appends entries here; pages/2_News_Queue.py
renders each as a card with one button.

Both `mechanism: "injury_score"` and `mechanism: "projection_pct"`
entries can now be applied directly from the queue page (2026-08-27,
user-directed -- projection edits used to require a manual code edit;
the review GATE this policy cares about is the human clicking Apply,
which the button still requires, not which file receives the write).
The page also lets the value be revised before applying, not just
accepted as proposed -- either in the app (edit the number, then Apply)
or in conversation with Claude (which rewrites the queue entry first).

A `projection_pct` entry patches whichever layer actually takes effect
for that player: model_projections_v1.csv if he has a row there (that
layer silently overrides PROJECTION_MANUAL_ADJUSTMENTS for anyone it
covers -- see the policy doc), else master_players.csv's
projection_points directly. Either way this bakes the new number into
the data file, the same "CSV patch is what makes it live today, dict
sync is later housekeeping" pattern §7 already uses for injury_score --
so if this player later ALSO gets a PROJECTION_MANUAL_ADJUSTMENTS dict
entry during a housekeeping pass, don't double-apply on top of an
already-baked-in number.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from draftkit.risk_scoring import load_weights, player_from_variable_row, risk_index

QUEUE_PATH = Path("research/pending_news_adjustments.json")
LOG_PATH = Path("research/applied_news_overrides_log.md")
RISK_CSV_PATH = Path("data/processed/risk_variables.csv")
MODEL_PROJECTIONS_PATH = Path("data/processed/model_projections_v1.csv")
MASTER_PLAYERS_PATH = Path("data/processed/master_players.csv")


def load_queue():
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def save_queue(entries):
    QUEUE_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def apply_injury_override(entry, score=None):
    """Patch risk_variables.csv the same way the real pipeline would,
    log it, and return (before, after) risk_index for the confirmation
    message. `score` overrides entry["value"] when the reviewer revised
    the number before applying. Raises if the player can't be found or
    the row is ambiguous."""
    value = entry["value"] if score is None else float(score)
    df = pd.read_csv(RISK_CSV_PATH)
    idx = df.index[df["player_name"] == entry["player"]]
    if len(idx) != 1:
        raise ValueError(f"{entry['player']}: expected 1 row in risk_variables.csv, found {len(idx)}")
    i = idx[0]
    before_score = df.at[i, "injury_score"]
    before_risk = df.at[i, "risk_index"]

    df.at[i, "injury_score"] = value
    if "injury_override_note" in df.columns:
        df.at[i, "injury_override_note"] = f"{entry['reason']} (reported {entry['date']})"

    weights = load_weights()
    player = player_from_variable_row(df.loc[i])
    after_risk = round(risk_index(player, weights), 1)
    df.at[i, "risk_index"] = after_risk
    df.to_csv(RISK_CSV_PATH, index=False)

    _log(f"- **{date.today().isoformat()}** -- APPLIED {entry['player']}: "
         f"injury_score {before_score} -> {value}, risk_index {before_risk} -> {after_risk}. "
         f"{entry['reason']}")
    return before_risk, after_risk


def resolve_projection_base(player_name):
    """Return (raw_base_points, current_effective_points, layer) for this
    player -- the RAW number a projection_pct edit multiplies against, and
    the number currently on the board.

    Real bug fixed 2026-08-27 (found live: Trevor Lawrence's Apply silently
    did nothing): this used to only check model_projection_points /
    model_projection_points_adjusted, but apply_model_projection_override()
    in draft_analysis.py -- the function that actually decides what
    projection_points shows on the board -- has a THIRD tier this one
    didn't know about: model_projection_points_fallback (a book-anchored
    or name-keyed placeholder value, used when no real
    MODEL_PROJECTION_CORRECTIONS entry exists yet). For any player in that
    third tier, model_projection_points/_adjusted are both NaN, so this
    returned raw=NaN. NaN is truthy in Python, so the caller's `if raw`
    guards didn't catch it either -- it silently propagated into a NaN
    pct, which apply_projection_override() then wrote into
    model_adjustment_pct. That NaN write is the actual reason the edit
    appeared to do nothing: apply_model_projection_override()'s own
    `.notna()` check on model_adjustment_pct now saw NaN, decided this
    row was "not corrected" after all, and fell back to the unchanged
    fallback value -- so the board kept showing the original number.

    Mirrors apply_model_projection_override()'s exact 3-tier priority so
    the two can never disagree on which number is "real" for a given
    player: (1) a real correction (model_adjustment_pct notna) -> raw is
    model_projection_points, current is model_projection_points_adjusted;
    (2) a fallback (model_projection_points_fallback notna, excluding the
    generic "Rookie-score fallback" composite placeholder that function
    also excludes -- see its own docstring for why) -> that fallback value
    is BOTH raw and current, since there's no separate "unadjusted model
    output" for a fallback-only row; (3) master_players.csv's raw
    projection_points.

    A percent change always applies to raw, never stacks on the current
    adjusted number. Shared by apply_projection_override() and
    preview_projection_rank() so the preview can never show a different
    number than Apply would actually produce. Raises if the player can't
    be found (or is ambiguous) in EITHER file, or if the resolved raw
    value is NaN (a real, still-unresolved gap upstream -- surfacing it
    is better than silently propagating NaN like the old code did)."""
    model_df = pd.read_csv(MODEL_PROJECTIONS_PATH) if MODEL_PROJECTIONS_PATH.exists() else pd.DataFrame()
    idx = model_df.index[model_df["player_name"] == player_name] if not model_df.empty else pd.Index([])

    if len(idx) > 1:
        raise ValueError(f"{player_name}: {len(idx)} rows in model_projections_v1.csv, expected 0 or 1")

    if len(idx) == 1:
        i = idx[0]
        pct = model_df.at[i, "model_adjustment_pct"] if "model_adjustment_pct" in model_df.columns else None
        if pd.notna(pct):
            raw = model_df.at[i, "model_projection_points"]
            current = model_df.at[i, "model_projection_points_adjusted"]
            # A fallback-tier player who has already been adjusted ONCE lands
            # here, not in the fallback branch below: the first edit wrote
            # model_adjustment_pct + model_projection_points_adjusted, but
            # model_projection_points stays NaN forever, because that column
            # is empty precisely because no real model output ever existed
            # for him (status insufficient_history / book_anchored -- the
            # whole reason he had a fallback). The pct was computed against
            # the FALLBACK, so that stays the raw base to keep multiplying
            # against. Reading model_projection_points unconditionally here
            # made every SECOND edit of such a player raise -- real bug,
            # found live 2026-08-28 on Jeremiyah Love (fallback 225.6, first
            # edit -1.5% -> 222.2, second edit errored). Only genuinely
            # inconsistent rows -- no raw AND no fallback to fall back on --
            # still raise.
            if pd.isna(raw) and "model_projection_points_fallback" in model_df.columns:
                raw = model_df.at[i, "model_projection_points_fallback"]
            if pd.isna(raw) or pd.isna(current):
                raise ValueError(
                    f"{player_name}: model_adjustment_pct is set but there is no usable base "
                    f"(model_projection_points and model_projection_points_fallback are both NaN) "
                    f"in model_projections_v1.csv -- inconsistent row, needs a data fix."
                )
            return float(raw), float(current), "model_projections_v1.csv"

        fallback = model_df.at[i, "model_projection_points_fallback"] if "model_projection_points_fallback" in model_df.columns else None
        fallback_note = model_df.at[i, "model_projection_fallback_note"] if "model_projection_fallback_note" in model_df.columns else None
        is_composite_placeholder = isinstance(fallback_note, str) and "Rookie-score fallback" in fallback_note
        if pd.notna(fallback) and not is_composite_placeholder:
            value = float(fallback)
            return value, value, "model_projections_v1.csv (fallback)"

    master_df = pd.read_csv(MASTER_PLAYERS_PATH, low_memory=False)
    midx = master_df.index[master_df["player_name"] == player_name]
    if len(midx) != 1:
        raise ValueError(f"{player_name}: expected 1 row in master_players.csv, found {len(midx)}")
    raw = master_df.at[midx[0], "projection_points"]
    if pd.isna(raw):
        raise ValueError(f"{player_name}: projection_points is NaN in master_players.csv -- nothing to adjust from.")
    return float(raw), float(raw), "master_players.csv"


def apply_projection_override(entry, pct=None):
    """Patch whichever projection layer actually takes effect for this
    player -- see module docstring. `pct` overrides entry["value"] when the
    reviewer revised the number before applying. Returns (before, after,
    layer) for the confirmation message."""
    value = entry["value"] if pct is None else float(pct)
    if value != value:  # NaN check without importing math -- NaN != NaN is always True
        raise ValueError(f"{entry['player']}: refusing to apply a NaN pct (upstream bug -- check the caller).")
    raw, before, layer = resolve_projection_base(entry["player"])
    after = round(raw * (1 + value / 100), 1)

    if layer.startswith("model_projections_v1.csv"):
        df = pd.read_csv(MODEL_PROJECTIONS_PATH)
        i = df.index[df["player_name"] == entry["player"]][0]
        df.at[i, "model_projection_points_adjusted"] = after
        df.at[i, "model_adjustment_pct"] = value
        df.at[i, "model_adjustment_note"] = (
            f"{entry['reason']} (applied via News Queue, {date.today().isoformat()})"
        )
        df.to_csv(MODEL_PROJECTIONS_PATH, index=False)
    else:
        df = pd.read_csv(MASTER_PLAYERS_PATH, low_memory=False)
        i = df.index[df["player_name"] == entry["player"]][0]
        df.at[i, "projection_points"] = after
        df.to_csv(MASTER_PLAYERS_PATH, index=False)

    _log(f"- **{date.today().isoformat()}** -- APPLIED {entry['player']}: "
         f"projection_pct {value:+.1f}% via {layer}, points {before} -> {after}. "
         f"{entry['reason']}")
    return before, after, layer


def preview_projection_rank(entry, pct, board_df):
    """Positional rank by projection_points right now vs. if `pct` were
    applied -- e.g. (position="RB", current_rank=50, new_rank=48, total=84).
    Uses resolve_projection_base() for the new-points math, the exact
    same computation Apply itself uses, so the preview can never drift
    from what clicking Apply would actually do. `board_df` is the live
    board (build_recommendation_rankings_df()), passed in rather than
    loaded here so callers can cache/reuse it across every card on the
    page instead of reloading per player. Returns None if the player or
    his position can't be found on the current board."""
    player_name = entry["player"]
    row = board_df[board_df["player_name"] == player_name]
    if row.empty:
        return None
    position = str(row.iloc[0]["position"]).upper()
    pos_df = board_df[board_df["position"].astype(str).str.upper() == position]
    if pos_df.empty:
        return None

    try:
        raw, current_pts, _ = resolve_projection_base(player_name)
    except ValueError:
        return None
    new_pts = round(raw * (1 + pct / 100), 1)

    pts = pos_df["projection_points"].astype(float)
    current_rank = int((pts > current_pts).sum()) + 1
    other_pts = pts[pos_df["player_name"] != player_name]
    new_rank = int((other_pts > new_pts).sum()) + 1
    return {
        "position": position, "current_rank": current_rank, "new_rank": new_rank,
        "total": len(pos_df), "current_pts": current_pts, "new_pts": new_pts,
    }


def dismiss_entry(entry, note="dismissed without changes"):
    _log(f"- **{date.today().isoformat()}** -- DISMISSED {entry['player']} "
         f"(proposed {entry.get('mechanism', '?')}={entry.get('value', '?')}). {note}")


def _log(line):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "# Applied / dismissed news overrides -- audit log\n\n"
                "Everything the resolve queue (pages/2_News_Queue.py) has acted on. "
                "Reversible: edit the risk_variables.csv row back, or re-run the "
                "opposite action.\n\n---\n\n"
            )
        f.write(line + "\n")
