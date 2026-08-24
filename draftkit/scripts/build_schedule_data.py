"""One-time pull: 2026 NFL schedule -> per-team schedule/weather/venue risk inputs.

New for risk scorecard v2's schedule_weather_venue category (v1 had no
schedule data at all). Source: nflverse-data's "schedules" release, the
same GitHub-releases pattern already used elsewhere in this repo (e.g.
research/validation_v1/build_offense_environment_features_v1.py). Verified
live before building this: the real asset is games.csv (a combined
all-seasons file, not a per-season file), confirmed to return 272 real
2026 REG-season games with roof/rest/weekday/gametime.

No true game-day temperature exists for future games (not knowable months
out -- confirmed by inspection, `temp` is NaN for all 2026 rows). `roof`
type is the real, available proxy for weather risk, matching the spec's
own "historically likely" framing rather than claiming forecast accuracy.

season_sos_rank has no true defensive-SOS data behind it (none exists in
this repo). Built as a documented proxy: average opponent
prior_team_epa_pg (offense_environment_team_seasons.csv) across a team's
FULL real season of opponents -- measures opponent OFFENSE strength as an
overall team-quality stand-in, not real defensive SOS. Labeled as a proxy
here and in every consumer, not presented as authoritative.

v5 (risk_engine_v5_master_fix_spec.pdf): home_climate (a team's own
stadium type, a constant) and road_cold_games (only true road trips into
a cold outdoor stadium) replace dome_games/outdoor_cold_games, which
incorrectly treated a team's own home dome games as risk-reducing and any
cold game (home or away) as risk-adding -- a team's home stadium type
isn't a risk factor, only unusual road environments are.

Post-v5 fix: this SOS proxy originally only averaged weeks 15-17
opponents ("fantasy playoff weeks"), a deliberate design choice from the
v2 build. Confirmed broken relative to real expectations: real-world
"easiest schedule in football" claims (Sharp Football and others) are
whole-season aggregates, and by that measure Cincinnati's real 2026
opponents are the WEAKEST average offense in the entire league (rank
32/32 by this same EPA proxy, computed across all games) while the
weeks-15-17-only version ranked them 15th -- middling, not close to
"easiest." Detroit shows the same pattern (24th whole-season vs. 21st
narrow). Widened to average across the team's full real schedule to match
what "schedule difficulty" actually means to users.

Usage:
    python -m draftkit.scripts.build_schedule_data
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
OFFENSE_CSV = REPO_ROOT / "research" / "validation_v1" / "data" / "offense_environment_team_seasons.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "team_schedule_risk.csv"

SEASON = 2026
DOME_ROOF = {"dome", "closed"}
OUTDOOR_ROOF = {"outdoors", "open"}
COLD_MONTHS = {11, 12, 1}
# Teams in genuinely cold-weather cities -- static geography, not projected.
COLD_WEATHER_TEAMS = {
    "GB", "BUF", "CHI", "CLE", "CIN", "PIT", "KC", "DEN",
    "MIN", "NE", "NYJ", "NYG", "PHI", "WAS", "BAL",
}
SHORT_WEEK_REST_DAYS = 6
PRIMETIME_WEEKDAYS = {"Thursday", "Monday"}
SNF_MIN_HOUR = 20


def fetch_games() -> pd.DataFrame:
    request = urllib.request.Request(GAMES_URL, headers={"User-Agent": "GuaranteedPlay/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    df = pd.read_csv(io.BytesIO(raw))
    games = df[(df["season"] == SEASON) & (df["game_type"] == "REG")].copy()
    games["gameday"] = pd.to_datetime(games["gameday"])
    games["month"] = games["gameday"].dt.month
    return games


def _team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) -- each game counted once for the home team
    and once for the away team, with the game's actual venue conditions
    (roof, and the STADIUM's team for weather purposes) attached to both.

    stadium_team is always the home team of that specific game -- distinct
    from `team` (which flips depending on whether this row represents the
    home or away side). Using `team` for the cold-weather check instead of
    stadium_team was an early bug caught by spot-checking: it made a warm
    weather team's road trip to a cold stadium never count as a cold game
    (confirmed via MIA showing outdoor_cold_games=0, which is wrong -- MIA
    plays road games at genuinely cold stadiums in Nov/Dec).
    """
    home = games.rename(columns={"home_team": "team", "away_team": "opponent", "home_rest": "rest"})
    home["stadium_team"] = home["team"]
    home = home[["team", "opponent", "stadium_team", "week", "roof", "month", "rest", "weekday", "gametime"]]
    away = games.rename(columns={"away_team": "team", "home_team": "opponent", "away_rest": "rest"})
    away["stadium_team"] = away["opponent"]
    away = away[["team", "opponent", "stadium_team", "week", "roof", "month", "rest", "weekday", "gametime"]]
    return pd.concat([home, away], ignore_index=True)


def _is_primetime(row) -> bool:
    if row["weekday"] in PRIMETIME_WEEKDAYS:
        return True
    if row["weekday"] == "Sunday" and isinstance(row["gametime"], str):
        try:
            hour = int(row["gametime"].split(":")[0])
        except (ValueError, IndexError):
            return False
        return hour >= SNF_MIN_HOUR
    return False


def compute_dome_and_weather(team_games: pd.DataFrame) -> pd.DataFrame:
    """v5: home_climate (a team's own stadium type -- constant, not a risk
    factor) and road_cold_games (only true road trips into a cold outdoor
    stadium) replace dome_games/outdoor_cold_games, which incorrectly
    counted a team's OWN home dome games as risk-REDUCING and any cold
    game (including home ones) as risk-adding. A team's home games are
    all at the same physical stadium in a season (barring rare neutral-
    site games, not modeled here), so the first home row's roof type is
    the team's real home_climate."""
    home_rows = team_games[team_games["team"] == team_games["stadium_team"]]
    home_climate = home_rows.drop_duplicates("team", keep="first").set_index("team")["roof"].map(
        lambda roof: "dome" if roof in DOME_ROOF else "outdoor"
    ).rename("home_climate")

    road_rows = team_games[team_games["team"] != team_games["stadium_team"]]
    road_cold = (
        road_rows["roof"].isin(OUTDOOR_ROOF)
        & road_rows["stadium_team"].isin(COLD_WEATHER_TEAMS)
        & road_rows["month"].isin(COLD_MONTHS)
    )
    road_cold_games = road_rows.assign(is_cold=road_cold).groupby("team")["is_cold"].sum().rename(
        "road_cold_games"
    )

    return pd.concat([home_climate, road_cold_games], axis=1).reset_index()


def compute_bye_weeks(team_games: pd.DataFrame) -> pd.DataFrame:
    all_weeks = set(range(1, 19))
    rows = []
    for team, group in team_games.groupby("team"):
        played = set(group["week"].unique())
        missing = sorted(all_weeks - played)
        rows.append({"team": team, "bye_week": missing[0] if missing else None})
    return pd.DataFrame(rows)


def compute_season_sos(team_games: pd.DataFrame, offense: pd.DataFrame) -> pd.DataFrame:
    """Whole-season opponent-strength proxy (post-v5 fix -- see module
    docstring for why this replaced a weeks-15-17-only version)."""
    epa_by_team = offense[offense["season"] == SEASON].set_index("team")["prior_team_epa_pg"]
    season_games = team_games.copy()
    season_games["opponent_epa"] = season_games["opponent"].map(epa_by_team)

    avg_opp_epa = season_games.groupby("team")["opponent_epa"].mean().reset_index()
    # Higher opponent EPA/play = tougher opponents = rank 1 (hardest), per
    # the spec's "lower rank = harder" convention.
    avg_opp_epa["season_sos_rank"] = avg_opp_epa["opponent_epa"].rank(
        ascending=False, method="first"
    ).astype(int)
    return avg_opp_epa[["team", "season_sos_rank"]]


def compute_short_week_primetime(team_games: pd.DataFrame) -> pd.DataFrame:
    is_short = team_games["rest"] < SHORT_WEEK_REST_DAYS
    is_prime = team_games.apply(_is_primetime, axis=1)
    out = team_games.assign(flag=(is_short | is_prime))
    return out.groupby("team")["flag"].sum().reset_index().rename(
        columns={"flag": "short_week_or_primetime_games"}
    )


def build_schedule_risk() -> pd.DataFrame:
    games = fetch_games()
    team_games = _team_game_rows(games)
    offense = pd.read_csv(OFFENSE_CSV)

    dome_weather = compute_dome_and_weather(team_games)
    byes = compute_bye_weeks(team_games)
    sos = compute_season_sos(team_games, offense)
    fatigue = compute_short_week_primetime(team_games)

    result = dome_weather.merge(byes, on="team").merge(sos, on="team", how="left").merge(
        fatigue, on="team"
    )
    return result.sort_values("team").reset_index(drop=True)


def main() -> int:
    board = build_schedule_risk()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(board)} team(s)")
    print(board.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
