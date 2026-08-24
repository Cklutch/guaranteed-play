"""Fantasy draft risk scorecard -- CLI (spec Section 7).

v2: risk_index is the composite of exactly four weighted categories
(injury, role_usage_td, offense_environment, schedule_weather_venue).
market (ADP mispricing) and volatility (week-to-week variance proxy) are
NOT part of that composite -- market is a value signal, not a risk signal;
volatility is a downstream symptom of role instability, not an independent
driver. Both are still shown, but as separate lines beneath the weighted
breakdown, never blended into Risk Index.

Interactive REPL, run standalone (no Streamlit):

    python -m draftkit.risk_cli

Commands (kept short -- these get typed under a draft clock):
    taken <player>   Mark player taken by an opponent.
    mine <player>    Mark player drafted by you.
    undo             Reverses the single most recent pick.
    check <player>   Risk Index + fit signal for a candidate.
    board [position] Top available players by ADP with Risk Index + fit.
    roster           Your roster with per-category risk exposure totals.
    help / quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.risk_constraints import (  # noqa: E402
    DraftPool,
    TeamRiskBudget,
    check_correlated_risk,
    draft_pick,
    dynamic_risk_ceiling,
    evaluate_pick_fit,
    normalize_name,
    undo_pick,
)
from draftkit.risk_scoring import (  # noqa: E402
    Player,
    Position,
    RiskCategory,
    load_weights,
    player_from_variable_row,
    risk_index,
)
from draftkit.scripts.build_risk_variables import OUTPUT_CSV  # noqa: E402

CATEGORY_LABELS = {
    "injury": "Injury",
    "role_usage_td": "Role/Usage/TD",
    "offense_environment": "Offense Env",
    "schedule_weather_venue": "Schedule/Wx",
}

# player_key -> {"value_signal_market": float|None, "volatility_diagnostic": float|None}
# Populated by load_pool(), read by print_player_risk()/cmd_roster() -- these
# two are NOT RiskCategory members (see module docstring), so they don't
# live on Player/RiskVariable the way the four weighted categories do.
ValueVolatility = dict[str, dict]


def load_pool() -> tuple[DraftPool, ValueVolatility]:
    if not OUTPUT_CSV.exists():
        print(
            f"ERROR: {OUTPUT_CSV} not found. Run "
            "'python -m draftkit.scripts.build_risk_variables' first.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    df = pd.read_csv(OUTPUT_CSV)
    players: dict[str, Player] = {}
    value_volatility: ValueVolatility = {}

    for _, row in df.iterrows():
        player = player_from_variable_row(row)
        if player is None:
            continue  # build script already restricts to QB/RB/WR/TE; skip defensively
        key = normalize_name(str(row["player_name"]))
        players[key] = player
        value_volatility[key] = {
            "value_signal_market": row.get("value_signal_market"),
            "volatility_diagnostic": row.get("volatility_diagnostic"),
        }

    return DraftPool(all_players=players), value_volatility


def format_fit_reason(candidate: Player, budget: TeamRiskBudget, fit: str) -> str:
    if fit == "red":
        flags = check_correlated_risk(candidate, budget)
        tripped = [CATEGORY_LABELS[c] for c, v in flags.items() if v]
        return f"Stacks {', '.join(tripped)} risk with a player already on your roster"
    if fit == "yellow":
        ceilings = dynamic_risk_ceiling(budget)
        tripped = [
            CATEGORY_LABELS[cat.value]
            for cat in RiskCategory
            if candidate.category_score(cat) > ceilings[cat.value]
        ]
        return f"{', '.join(tripped)} risk near ceiling given your current {candidate.position.value} picks"
    return "No red flags"


def _format_extra(label: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"  {label}: n/a"
    return f"  {label}: {value:+.2f}" if isinstance(value, float) else f"  {label}: {value}"


def print_player_risk(
    player: Player, weights: dict, value_volatility: ValueVolatility,
    budget: TeamRiskBudget | None = None,
) -> None:
    idx = risk_index(player, weights)
    parts = " ".join(
        f"{CATEGORY_LABELS[cat.value]}: {player.category_score(cat):.1f}" for cat in RiskCategory
    )
    print(f"{player.name} ({player.position.value}) -- Risk Index: {idx}")
    print(f"  {parts}")

    extra = value_volatility.get(normalize_name(player.name), {})
    print(_format_extra("Value signal (market, not in Risk Index)", extra.get("value_signal_market")))
    print(_format_extra("Volatility (diagnostic, not in Risk Index)", extra.get("volatility_diagnostic")))

    if budget is not None:
        fit = evaluate_pick_fit(player, budget)
        reason = format_fit_reason(player, budget, fit)
        print(f"  Fit signal: {fit.upper()} ({reason})")


def cmd_taken(pool: DraftPool, budget: TeamRiskBudget, args: str) -> None:
    try:
        player = draft_pick(pool, budget, args, by_me=False)
        print(f"Marked TAKEN_BY_OTHER: {player.name}")
    except ValueError as exc:
        print(f"ERROR: {exc}")


def cmd_mine(pool: DraftPool, budget: TeamRiskBudget, args: str) -> None:
    try:
        player = draft_pick(pool, budget, args, by_me=True)
        print(f"Drafted: {player.name} -> added to your roster and Team Risk Budget")
    except ValueError as exc:
        print(f"ERROR: {exc}")


def cmd_undo(pool: DraftPool, budget: TeamRiskBudget) -> None:
    key = undo_pick(pool, budget)
    if key is None:
        print("Nothing to undo.")
        return
    print(f"Undone: {pool.all_players[key].name}")


def cmd_check(
    pool: DraftPool, budget: TeamRiskBudget, weights: dict, value_volatility: ValueVolatility, args: str,
) -> None:
    try:
        key = pool.resolve_name(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return
    if pool.status[key].value != "available":
        print(f"ERROR: {pool.all_players[key].name} is already marked {pool.status[key].value}")
        return
    print_player_risk(pool.all_players[key], weights, value_volatility, budget)


def cmd_board(pool: DraftPool, weights: dict, args: str) -> None:
    position = None
    if args.strip():
        try:
            position = Position(args.strip().upper())
        except ValueError:
            print(f"ERROR: unknown position '{args.strip()}' (use QB/RB/WR/TE)")
            return
    available = sorted(pool.available_players(position), key=lambda p: p.adp)[:20]
    if not available:
        print("No available players match.")
        return
    for player in available:
        idx = risk_index(player, weights)
        print(f"  ADP {player.adp:>6.1f}  {player.position.value:3s}  {player.name:28s}  Risk Index: {idx}")


def cmd_roster(budget: TeamRiskBudget, weights: dict, value_volatility: ValueVolatility) -> None:
    if not budget.picks:
        print("No players drafted yet.")
        return
    print(f"My Team ({len(budget.picks)} players):")
    for player in budget.picks:
        idx = risk_index(player, weights)
        print(f"  {player.position.value:3s} {player.name:28s} Risk Index: {idx}")
    print("\nPer-category exposure (average across roster):")
    for category in RiskCategory:
        scores = budget.exposure_by_category(category)
        avg = sum(scores) / len(scores) if scores else 0.0
        high_count = budget.category_high_risk_count(category)
        print(f"  {CATEGORY_LABELS[category.value]:16s} avg={avg:.2f}  high-risk picks={high_count}")

    market_values = [
        value_volatility.get(normalize_name(p.name), {}).get("value_signal_market") for p in budget.picks
    ]
    market_values = [v for v in market_values if v is not None and pd.notna(v)]
    if market_values:
        print(f"\nValue signal (market) avg across roster: {sum(market_values)/len(market_values):+.2f}"
              " (not in Risk Index)")


def run(pool: DraftPool, value_volatility: ValueVolatility) -> int:
    weights = load_weights()
    budget = TeamRiskBudget()

    print("Fantasy Draft Risk Scorecard -- type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        command, _, args = line.partition(" ")
        command = command.lower()
        args = args.strip()

        if command in {"quit", "exit"}:
            return 0
        if command == "help":
            print(__doc__)
        elif command == "taken":
            cmd_taken(pool, budget, args)
        elif command == "mine":
            cmd_mine(pool, budget, args)
        elif command == "undo":
            cmd_undo(pool, budget)
        elif command == "check":
            cmd_check(pool, budget, weights, value_volatility, args)
        elif command == "board":
            cmd_board(pool, weights, args)
        elif command == "roster":
            cmd_roster(budget, weights, value_volatility)
        else:
            print(f"Unknown command: {command} (type 'help')")


def main() -> int:
    pool, value_volatility = load_pool()
    return run(pool, value_volatility)


if __name__ == "__main__":
    raise SystemExit(main())
