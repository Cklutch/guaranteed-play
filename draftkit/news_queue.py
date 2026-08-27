"""Light resolve queue for the news-override pipeline (Lane B / human
review -- see research/news_override_policy.md). The scheduled
nfl-camp-news-watch sweep appends entries here; pages/2_News_Queue.py
renders each as a card with one button.

Only `mechanism: "injury_score"` entries can be applied directly --
patching data/processed/risk_variables.csv is a data edit, safe to do
from a button. A `mechanism: "projection_pct"` entry needs a source-code
edit to PROJECTION_MANUAL_ADJUSTMENTS (draft_analysis.py), which this
module deliberately does NOT do automatically -- that lever is reserved
for a human/Claude to write, per policy §1a's higher bar for it.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from draftkit.risk_scoring import load_weights, player_from_variable_row, risk_index

QUEUE_PATH = Path("research/pending_news_adjustments.json")
LOG_PATH = Path("research/applied_news_overrides_log.md")
RISK_CSV_PATH = Path("data/processed/risk_variables.csv")


def load_queue():
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def save_queue(entries):
    QUEUE_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def apply_injury_override(entry):
    """Patch risk_variables.csv the same way the real pipeline would,
    log it, and return (before, after) risk_index for the confirmation
    message. Raises if the player can't be found or the row is ambiguous."""
    df = pd.read_csv(RISK_CSV_PATH)
    idx = df.index[df["player_name"] == entry["player"]]
    if len(idx) != 1:
        raise ValueError(f"{entry['player']}: expected 1 row in risk_variables.csv, found {len(idx)}")
    i = idx[0]
    before_score = df.at[i, "injury_score"]
    before_risk = df.at[i, "risk_index"]

    df.at[i, "injury_score"] = entry["value"]
    if "injury_override_note" in df.columns:
        df.at[i, "injury_override_note"] = f"{entry['reason']} (reported {entry['date']})"

    weights = load_weights()
    player = player_from_variable_row(df.loc[i])
    after_risk = round(risk_index(player, weights), 1)
    df.at[i, "risk_index"] = after_risk
    df.to_csv(RISK_CSV_PATH, index=False)

    _log(f"- **{date.today().isoformat()}** -- APPLIED {entry['player']}: "
         f"injury_score {before_score} -> {entry['value']}, risk_index {before_risk} -> {after_risk}. "
         f"{entry['reason']}")
    return before_risk, after_risk


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
