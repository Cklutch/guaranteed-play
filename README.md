# Guaranteed Play

A fantasy football draft engine that builds its own player valuations from
scratch instead of importing consensus rankings — plus a live in-draft
assistant that reasons about pick-survival odds and roster construction in
real time. Built to replace "trust the expert rankings" with a scoring
model I could actually audit, test, and improve.

<!--
TODO: drop a real screenshot or short GIF here before sharing this repo.
Run `.venv/Scripts/streamlit.exe run Home.py --server.headless true`,
screenshot the rankings table (tier bands + player cards are the best
shot), save it as docs/screenshot.png, then uncomment:
![Rankings board](docs/screenshot.png)
-->

**[Explore the architecture — interactive schematic](https://claude.ai/code/artifact/114fd273-eb1d-49cc-875e-ab91c71a7c36)** ·
a versioned copy also lives at [`docs/draft_engine_schematic.html`](docs/draft_engine_schematic.html)

## What makes this more than a rankings scraper

- **A weighted, sensitivity-tested scoring model.** `final_score` blends
  value-over-replacement (computed from the actual league roster format,
  not a generic assumption), a within-position projection percentile,
  a sportsbook-vs-ADP market signal, and a risk penalty. The four weights
  were stress-tested across a full sensitivity grid and pinned in a
  regression test that re-validates them against the live player pool on
  every run — not just checked once and forgotten.
- **A tier system that adapts instead of needing re-tuning.** Tiers are
  placed at the largest real gaps in the score distribution, not a fixed
  point threshold — so the board self-calibrates every time projections
  change instead of drifting into 19 fragmented micro-tiers.
- **A provably consistent survival model.** The in-draft assistant answers
  "how likely is this player still there at my next pick?" three different
  ways — unconditional, conditional on opponent picks only, and adjusted
  for what other teams still need — and a test suite proves algebraically
  that the simpler views are exact reductions of the richer one, not just
  similar-looking approximations.
- **Real bugs, found by testing and fixed on both sides.** The pick engine
  once ranked a strictly worse quarterback above a clearly better one,
  because scarcity math designed for *complementary* picks (you want both)
  was misapplied to a *substitute* pick (you'll only roster one). Fixed,
  and regression-tested in both directions so the fix couldn't quietly
  break the opposite case.
- **A research pipeline that's honest about what doesn't work.** Every new
  signal — a breakout-probability model, a rewritten championship-equity
  score — gets backtested against real historical outcomes before it's
  allowed near the live board. The championship-equity model was backtested
  against 14,771 real player-seasons and found to perform close to a coin
  flip (AUC 0.538); that result is documented plainly in the repo, not
  hidden, because knowing what *doesn't* work is part of the job.
- **A live news pipeline with an audit trail.** Real-world news (injuries,
  suspensions, depth-chart shifts) flows into live projections through a
  two-lane review process — low-stakes changes auto-apply, anything that
  moves a player's rank waits for a human, and every applied change is
  logged.

## Tech stack

Python · Streamlit · pandas · scikit-learn (research pipeline) · fully
server-rendered HTML/CSS for the UI (no client-side JS framework) ·
nflverse for historical play-by-play data · Sleeper API for live draft
sync.

## Architecture

Three layers: raw and processed data, a research pipeline that validates
signals before anything ships, and a scoring/interface layer built on top
of what survives validation. The interactive schematic linked above is the
fastest way to see how any one piece connects to the rest — click a module
to trace exactly what feeds it and what it feeds. [`CLAUDE.md`](CLAUDE.md)
has the same picture in prose, plus the sharper edge cases.

## Setup

```bash
git clone https://github.com/Cklutch/guaranteed-play.git
cd guaranteed-play
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/streamlit.exe run Home.py --server.headless true
```

Opens at `http://localhost:8501`. Two optional token files
(`.odds_api.token`, `.github_token`) are needed only for the offline data
scripts — the live app never calls either.
