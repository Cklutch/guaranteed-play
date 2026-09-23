"""In-season weekly value: blend real production with the preseason projection.

The rest of the app is preseason -- one full-season projection per player,
computed before a game is played. Once the season starts, that projection is
only half the story: what a player has *actually* done matters more with every
week, but three games is still a small sample you don't want to overreact to.

This module produces a rest-of-season (ROS) value by blending the two, using
real weekly scoring pulled from Sleeper's public stats API (the same ecosystem
as the live-draft sync, no auth):

    blended_ppg = w * actual_ppg + (1 - w) * preseason_ppg
    w           = games_played / (games_played + STABILIZER)

`w` is a shrinkage weight, not a hard switch: at 0 games it is 0 (pure
preseason), and it climbs toward 1 as real games accumulate, so the board
leans on actuals exactly as fast as the sample earns it. STABILIZER sets how
many games it takes to reach a 50/50 blend.

ROS points = blended_ppg * remaining_games, which is the currency the trade
calculator values a roster in -- preseason draft value is the wrong number for
a Week 3 trade (a fast riser or a bust has already moved off it).

Endpoints (https://api.sleeper.app/v1):
  * /state/nfl                       -- current week / season
  * /stats/nfl/regular/<season>/<wk> -- per-player actuals, keyed by Sleeper id

Only completed weeks carry data; the in-progress week returns {} and is
skipped, so this is always "through the last finished week."
"""

import pandas as pd
import requests

SLEEPER_BASE = "https://api.sleeper.app/v1"
_TIMEOUT = 15
_HEADERS = {"User-Agent": "guaranteed-play-draftkit"}

SEASON_GAMES = 17
STABILIZER = 4.0  # games to reach a 50/50 actuals-vs-preseason blend

SCORING_PTS_KEY = {
    "half_ppr": "pts_half_ppr",
    "ppr": "pts_ppr",
    "std": "pts_std",
    "standard": "pts_std",
}


def _get(url):
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_state(default_week=1, default_season="2026"):
    """Current NFL week + season from Sleeper. Robust to any fetch error."""
    try:
        s = _get(f"{SLEEPER_BASE}/state/nfl")
        return {
            "week": int(s.get("week") or default_week),
            "season": str(s.get("season") or default_season),
        }
    except Exception:
        return {"week": default_week, "season": default_season}


def _pid_key(pid):
    """Normalize a pool player_id to the string form Sleeper stats use.

    Pool ids arrive as floats (8112.0); Sleeper keys are strings ("8112").
    """
    if pd.isna(pid):
        return None
    key = str(pid)
    return key[:-2] if key.endswith(".0") else key


def fetch_weekly_actuals(season, week):
    """{sleeper_player_id: stat_dict} for one completed week; {} if not scored."""
    try:
        data = _get(f"{SLEEPER_BASE}/stats/nfl/regular/{season}/{week}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def actuals_to_date(season, through_week, scoring="half_ppr"):
    """Per-player real production over completed weeks 1..through_week.

    Returns {player_id: {"games": int, "points": float, "ppg": float}} using
    only weeks that actually have scores (an unplayed week contributes nothing).
    """
    pts_key = SCORING_PTS_KEY.get(scoring, "pts_half_ppr")
    totals = {}
    for wk in range(1, max(through_week, 0) + 1):
        week_data = fetch_weekly_actuals(season, wk)
        if not week_data:
            continue
        for pid, stat in week_data.items():
            if not isinstance(stat, dict):
                continue
            pts = stat.get(pts_key)
            if pts is None:
                continue
            # gp guards against players who were rostered but did not play.
            played = float(stat.get("gp") or 0) > 0
            agg = totals.setdefault(pid, {"games": 0, "points": 0.0})
            agg["points"] += float(pts)
            if played:
                agg["games"] += 1
    for agg in totals.values():
        agg["ppg"] = agg["points"] / agg["games"] if agg["games"] else 0.0
    return totals


def build_week_values(
    board_df,
    scoring="half_ppr",
    season=None,
    current_week=None,
    stabilizer=STABILIZER,
    season_games=SEASON_GAMES,
    projection_col="projection_points",
):
    """Add in-season ROS columns to a copy of the board and re-rank by them.

    New columns:
      games_played   -- completed games with real data
      actual_ppg     -- real points per game so far (0 if no games yet)
      preseason_ppg  -- the preseason full-season projection / season_games
      blend_weight   -- w, how far the blend leans on actuals (0..1)
      blended_ppg    -- w*actual + (1-w)*preseason
      ros_points     -- blended_ppg * remaining games (the trade currency)
      week_rank      -- overall rank by ros_points (1 = best)

    Degrades safely: if the stats fetch is empty (offseason, API down), every
    blend_weight is 0 and ros_points is just the preseason projection pro-rated
    for the weeks remaining -- i.e. the board falls back to preseason, never
    errors.
    """
    board = board_df.copy()

    state = get_state()
    season = season or state["season"]
    current_week = current_week if current_week is not None else state["week"]
    weeks_elapsed = max(current_week - 1, 0)
    remaining_games = max(season_games - weeks_elapsed, 1)

    actuals = actuals_to_date(season, weeks_elapsed, scoring)

    proj = pd.to_numeric(board.get(projection_col), errors="coerce").fillna(0.0)
    preseason_ppg = proj / season_games

    games_played, actual_ppg = [], []
    for pid in board.get("player_id", pd.Series([None] * len(board))):
        rec = actuals.get(_pid_key(pid))
        if rec:
            games_played.append(rec["games"])
            actual_ppg.append(rec["ppg"])
        else:
            games_played.append(0)
            actual_ppg.append(0.0)

    board["games_played"] = games_played
    board["actual_ppg"] = actual_ppg
    board["preseason_ppg"] = preseason_ppg.values

    gp = board["games_played"].astype(float)
    w = gp / (gp + float(stabilizer))
    # A player with zero games keeps pure preseason (w already 0 there).
    board["blend_weight"] = w
    board["blended_ppg"] = w * board["actual_ppg"] + (1.0 - w) * board["preseason_ppg"]
    board["ros_points"] = board["blended_ppg"] * remaining_games

    board = board.sort_values("ros_points", ascending=False).reset_index(drop=True)
    board["week_rank"] = range(1, len(board) + 1)

    board.attrs["in_season_meta"] = {
        "season": season,
        "current_week": current_week,
        "weeks_scored": weeks_elapsed,
        "remaining_games": remaining_games,
        "scoring": scoring,
        "players_with_actuals": int((board["games_played"] > 0).sum()),
    }
    return board


def build_in_season_board(
    master_board,
    scoring="half_ppr",
    season=None,
    current_week=None,
    season_games=SEASON_GAMES,
):
    """Re-score the board on in-season value, through the Base Value engine.

    Ranking on raw ros_points reintroduces the cross-position scoring-scale
    bias the preseason engine was built to remove (a QB outscores every RB on
    raw points). So instead of ranking on points, this threads the blended
    in-season pace back through Base Value's own position-relative machinery --
    the same VOR replacement baselines and within-position projection
    percentile the preseason board uses -- so a Week 3 board is position-fair
    for exactly the same reason the preseason one is.

    Takes the already-built master recommendations frame (which carries `adp`,
    `injury_risk`, `position`), never the raw CSV, so the market and risk
    components are the real ones. Returns a NEW frame; the preseason board it
    was handed is untouched, which is what keeps the two views independent and
    both on file to toggle between.

    New/overwritten columns beyond build_week_values':
      is_projection   -- blended_ppg * season_games (full-season-equivalent pace)
      value_over_replacement_points, projection_points -- recomputed on that pace
      base_value_* / base_value_score -- the four components, position-aware
      in_season_rank  -- overall rank by the in-season base_value_score
    """
    from draftkit.draft_analysis import (
        build_position_replacement_baselines,
        calculate_position_value_score,
        calculate_base_value_score,
    )

    board_in = master_board
    # build_recommendation_rankings_df() drops player_id, but the Sleeper stats
    # join is id-keyed. Both frames derive from master_players.csv with the same
    # name spelling, so restore the id by an exact-name merge when it's absent.
    if "player_id" not in board_in.columns or board_in["player_id"].isna().all():
        from draftkit.data_access import load_players_df

        players = load_players_df()
        if players is not None and {"player_name", "player_id"}.issubset(players.columns):
            id_map = (
                players[["player_name", "player_id"]]
                .dropna(subset=["player_name"])
                .drop_duplicates("player_name")
            )
            board_in = board_in.merge(id_map, on="player_name", how="left")

    board = build_week_values(
        board_in,
        scoring=scoring,
        season=season,
        current_week=current_week,
        season_games=season_games,
    )
    meta = board.attrs.get("in_season_meta", {})

    # Full-season-equivalent pace keeps VOR baselines on the same scale the
    # engine expects (points over a season), so replacement math stays sensible.
    board["is_projection"] = board["blended_ppg"] * season_games

    baselines = build_position_replacement_baselines(board, "position", "is_projection")
    board["value_over_replacement_points"] = [
        calculate_position_value_score(proj, pos, baselines)
        for proj, pos in zip(board["is_projection"], board["position"])
    ]

    # Swap the projection the engine reads to the in-season pace, then let the
    # unmodified Base Value function do the position-relative percentile itself.
    board["preseason_projection_points"] = board["projection_points"]
    board["projection_points"] = board["is_projection"]
    board = calculate_base_value_score(board)

    board = board.sort_values("base_value_score", ascending=False).reset_index(drop=True)
    board["in_season_rank"] = range(1, len(board) + 1)
    board.attrs["in_season_meta"] = meta
    return board
