# Guaranteed Play

2026 fantasy football draft rankings and pick-recommendation tool, built on a
custom Base Value scoring engine (VOR + projection + market signal + risk),
rendered as a Streamlit app.

## Setup on a new machine

```bash
git clone https://github.com/Cklutch/guaranteed-play.git
cd guaranteed-play
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Copy these two files to the repo root manually — they're gitignored and never
committed:
- `.odds_api.token` — required to run `draftkit/scripts/pull_sportsbook_props.py`
  (or set env var `THE_ODDS_API_KEY`)
- `.github_token` — optional, raises the nflverse GitHub API rate limit
  (or set env var `NFLVERSE_GITHUB_TOKEN` / `GITHUB_TOKEN`)

## Run

```bash
.venv/Scripts/streamlit.exe run Home.py --server.headless true
```

Opens at `http://localhost:8501`.

See [CLAUDE.md](CLAUDE.md) for architecture, the scoring model, and known
gotchas.
