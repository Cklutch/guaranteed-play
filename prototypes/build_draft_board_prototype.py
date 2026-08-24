"""Generate a static design mock of a tier-grid draft board.

DESIGN PROTOTYPE ONLY -- this is not wired into the app. It reads the real
board data so the mock can be judged on real players/tiers/ADP rather than
placeholder text, and emits one self-contained HTML file with inline CSS.

The emitted markup/CSS is meant to be reusable in whichever implementation
path gets chosen afterwards (Streamlit + custom CSS, or a React component).

Usage:
    python prototypes/build_draft_board_prototype.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run directly (python prototypes/build_draft_board_prototype.py), so the repo
# root isn't on sys.path the way it is for `python -m draftkit.scripts.*`.
sys.path.insert(0, str(REPO_ROOT))

from draftkit.projection_enrichment import normalize_player_name  # noqa: E402

MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
SPORTSBOOK_CSV = REPO_ROOT / "data" / "processed" / "sportsbook_vs_adp_comparison.csv"
OUTPUT_HTML = Path(__file__).resolve().parent / "draft_board_prototype.html"

BOARD_SIZE = 120          # players shown across the tier columns
DRAFTED_DEMO_RANKS = {1, 3, 6, 9}  # pre-marked "drafted" purely to show the struck-through style

POSITION_COLORS = {
    "QB": "#22c55e",
    "RB": "#ef4444",
    "WR": "#3b82f6",
    "TE": "#a855f7",
}
DEFAULT_POSITION_COLOR = "#64748b"


def load_board() -> pd.DataFrame:
    master = pd.read_csv(MASTER_CSV)
    master = master[master["adp_rank"].notna()].copy()
    master = master.sort_values("adp_rank").head(BOARD_SIZE)
    master["_key"] = master["player_name"].apply(normalize_player_name)

    if SPORTSBOOK_CSV.exists():
        book = pd.read_csv(SPORTSBOOK_CSV)
        keep = [
            c for c in ("player_name", "adp_position_rank", "sportsbook_position_rank",
                        "position_rank_gap")
            if c in book.columns
        ]
        book = book[keep].copy()
        book["_key"] = book["player_name"].apply(normalize_player_name)
        book = book.drop(columns=["player_name"]).drop_duplicates("_key")
        master = master.merge(book, on="_key", how="left")

    return master


def value_badge(gap) -> tuple[str, str] | None:
    """Green when the sportsbook ranks him better within position than ADP does."""
    if pd.isna(gap):
        return None
    gap = float(gap)
    if gap >= 5:
        return ("Market Value", "badge-value")
    if gap <= -5:
        return ("Market Fade", "badge-fade")
    return None


def render_card(row, overall_rank: int) -> str:
    position = str(row.get("position") or "").upper()
    color = POSITION_COLORS.get(position, DEFAULT_POSITION_COLOR)
    name = html.escape(str(row.get("player_name") or ""))
    team = html.escape(str(row.get("team") or "FA"))
    adp = row.get("adp")
    adp_text = f"ADP {float(adp):.1f}" if pd.notna(adp) else "ADP --"

    # Stored in the comparison CSV as position-prefixed labels ("RB1"), not
    # bare numbers -- Home.py strips the prefix at read time, we keep it here.
    book_rank = row.get("sportsbook_position_rank")
    book_text = ""
    if pd.notna(book_rank):
        book_text = f'<span class="book">BOOK {html.escape(str(book_rank))}</span>'

    badge = value_badge(row.get("position_rank_gap"))
    badge_html = ""
    if badge:
        label, cls = badge
        badge_html = f'<span class="badge {cls}">{label}</span>'

    drafted = " drafted" if overall_rank in DRAFTED_DEMO_RANKS else ""

    return f"""        <div class="card{drafted}" style="--pos-color:{color}">
          {badge_html}
          <div class="card-top">
            <span class="pos" style="background:{color}">{position}</span>
            <span class="name">{name}</span>
            <span class="rank">#{overall_rank}</span>
          </div>
          <div class="card-bottom">
            <span class="team">{team}</span>
            <span class="adp">{adp_text}</span>
            {book_text}
          </div>
        </div>"""


def render_tier_column(tier, rows) -> str:
    tier_label = "Unranked" if pd.isna(tier) else f"Tier {int(tier)}"
    cards = "\n".join(render_card(row, int(row["_overall"])) for _, row in rows.iterrows())
    return f"""      <section class="tier">
        <header class="tier-head">
          <h2>{tier_label}</h2>
          <span>{len(rows)} players</span>
        </header>
{cards}
      </section>"""


def render_side_player(row, overall_rank: int, primary: bool = False) -> str:
    position = str(row.get("position") or "").upper()
    color = POSITION_COLORS.get(position, DEFAULT_POSITION_COLOR)
    name = html.escape(str(row.get("player_name") or ""))
    team = html.escape(str(row.get("team") or "FA"))
    adp = row.get("adp")
    adp_text = f"ADP {float(adp):.1f}" if pd.notna(adp) else "ADP --"
    proj = row.get("projection_points")
    proj_text = f"{float(proj):.0f} pts" if pd.notna(proj) else ""

    if primary:
        return f"""      <div class="best-pick" style="--pos-color:{color}">
        <div class="kicker">★ Best Pick</div>
        <div class="best-name">{name}</div>
        <div class="best-meta"><span class="pos" style="background:{color}">{position}</span> {team} · {adp_text} · {proj_text}</div>
        <div class="best-note">Premium RB — 12-team RB scarcity</div>
      </div>"""

    return f"""        <div class="next-row">
          <span class="pos" style="background:{color}">{position}</span>
          <span class="next-name">{name}</span>
          <span class="next-rank">#{overall_rank}</span>
        </div>"""


def build_html(board: pd.DataFrame) -> str:
    board = board.reset_index(drop=True)
    board["_overall"] = board.index + 1

    tier_columns = []
    for tier, rows in board.groupby("tier", dropna=False, sort=True):
        tier_columns.append(render_tier_column(tier, rows))
    tiers_html = "\n".join(tier_columns)

    best = board.iloc[0]
    next_best = board.iloc[1:4]
    next_html = "\n".join(
        render_side_player(row, int(row["_overall"])) for _, row in next_best.iterrows()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draft Board — Design Prototype</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #0b0f16; color: #e6edf3;
    font: 13px/1.4 -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  .topbar {{
    display: flex; align-items: center; gap: 24px;
    padding: 10px 18px; background: #0f1621; border-bottom: 1px solid #1e2836;
    font-size: 12px; position: sticky; top: 0; z-index: 5;
  }}
  .topbar .live {{ color: #22c55e; font-weight: 700; letter-spacing: .04em; }}
  .topbar .muted {{ color: #7d8ea3; }}
  .topbar .need {{ color: #f59e0b; font-weight: 700; }}
  .layout {{ display: grid; grid-template-columns: 244px 1fr; height: calc(100vh - 39px); }}
  .rail {{ background: #0f1621; border-right: 1px solid #1e2836; padding: 14px; overflow-y: auto; }}
  .rail h1 {{ font-size: 15px; color: #f59e0b; margin: 0 0 14px; }}
  .best-pick {{
    background: #131c29; border: 1px solid #24303f; border-left: 3px solid var(--pos-color);
    border-radius: 6px; padding: 10px; margin-bottom: 16px;
  }}
  .kicker {{ font-size: 10px; color: #f59e0b; font-weight: 700; letter-spacing: .06em; }}
  .best-name {{ font-size: 16px; font-weight: 700; margin: 4px 0; }}
  .best-meta {{ font-size: 11px; color: #9fb0c3; display: flex; align-items: center; gap: 6px; }}
  .best-note {{ font-size: 11px; color: #7d8ea3; margin-top: 6px; }}
  .rail h3 {{ font-size: 10px; color: #7d8ea3; letter-spacing: .08em; margin: 0 0 8px; }}
  .next-row {{
    display: flex; align-items: center; gap: 7px;
    padding: 7px 0; border-bottom: 1px solid #1a2432;
  }}
  .next-name {{ flex: 1; font-size: 12px; }}
  .next-rank {{ color: #7d8ea3; font-size: 11px; }}
  .main {{ padding: 14px 18px; overflow: auto; }}
  .filters {{ display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }}
  .chip {{
    padding: 5px 13px; border-radius: 14px; background: #131c29;
    border: 1px solid #24303f; color: #9fb0c3; font-size: 12px;
  }}
  .chip.on {{ background: #f59e0b; color: #10151d; border-color: #f59e0b; font-weight: 700; }}
  .board {{ display: flex; gap: 12px; align-items: flex-start; overflow-x: auto; padding-bottom: 16px; }}
  .tier {{ flex: 0 0 208px; }}
  .tier-head {{
    background: #131c29; border: 1px solid #24303f; border-radius: 6px;
    padding: 7px 10px; margin-bottom: 8px;
  }}
  .tier-head h2 {{ font-size: 11px; margin: 0; letter-spacing: .07em; color: #c9d6e4; }}
  .tier-head span {{ font-size: 10px; color: #7d8ea3; }}
  .card {{
    position: relative; background: #131c29; border: 1px solid #24303f;
    border-left: 3px solid var(--pos-color); border-radius: 6px;
    padding: 8px 9px; margin-bottom: 6px;
  }}
  .card-top {{ display: flex; align-items: center; gap: 6px; }}
  .pos {{
    font-size: 9px; font-weight: 700; color: #0b0f16;
    padding: 1px 5px; border-radius: 3px;
  }}
  .name {{ flex: 1; font-size: 12px; font-weight: 600; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; }}
  .rank {{ font-size: 11px; color: #7d8ea3; font-weight: 700; }}
  .card-bottom {{
    display: flex; gap: 8px; margin-top: 4px; font-size: 10px; color: #7d8ea3;
  }}
  .book {{ color: #8b9bb0; }}
  .badge {{
    position: absolute; top: -6px; right: 6px;
    font-size: 8px; font-weight: 700; padding: 1px 5px; border-radius: 3px;
    letter-spacing: .03em;
  }}
  .badge-value {{ background: #16a34a; color: #f0fdf4; }}
  .badge-fade {{ background: #ea580c; color: #fff7ed; }}
  .card.drafted {{ opacity: .35; }}
  .card.drafted .name {{ text-decoration: line-through; }}
  .note {{ color: #7d8ea3; font-size: 11px; padding: 4px 18px 18px; }}
</style>
</head>
<body>
  <div class="topbar">
    <span class="live">● DESIGN PROTOTYPE</span>
    <span>Rd 1.12 · Pick 12</span>
    <span class="muted">Next: 2.11 · in 11</span>
    <span>Need: <span class="need">WR</span></span>
    <span class="muted">Static mock — not wired to draft state</span>
  </div>
  <div class="layout">
    <aside class="rail">
      <h1>Draft Day Cheat Sheet</h1>
{render_side_player(best, 1, primary=True)}
      <h3>NEXT BEST</h3>
{next_html}
    </aside>
    <div class="main">
      <div class="filters">
        <span class="chip on">All</span>
        <span class="chip">QB</span>
        <span class="chip">RB</span>
        <span class="chip">WR</span>
        <span class="chip">TE</span>
      </div>
      <div class="board">
{tiers_html}
      </div>
    </div>
  </div>
  <div class="note">
    Real data from master_players.csv + sportsbook_vs_adp_comparison.csv.
    Badges: <strong>Market Value</strong> = sportsbook ranks him 5+ spots better
    within position than ADP; <strong>Market Fade</strong> = 5+ spots worse.
    Dimmed/struck rows are demo-only to show the drafted state.
  </div>
</body>
</html>
"""


def main() -> int:
    board = load_board()
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(build_html(board), encoding="utf-8")

    tiers = board["tier"].nunique(dropna=False)
    badged = board["position_rank_gap"].notna().sum() if "position_rank_gap" in board.columns else 0
    print(f"[write] {OUTPUT_HTML}")
    print(f"players: {len(board)} | tier columns: {tiers} | rows with sportsbook gap: {badged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
