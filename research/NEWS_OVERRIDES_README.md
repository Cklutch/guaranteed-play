# News overrides — how this works

The board's projections and risk scores come from a data pipeline that
only refreshes occasionally (see CLAUDE.md's Gotchas). Real news — an
injury, a suspension review, a camp depth-chart shakeup — is often true
*today* but won't reach the board until the next full rebuild, which can
be weeks away. This system exists to close that gap: catch real news,
turn it into a small, disclosed, reviewable adjustment, without waiting
on a rebuild.

## The lifecycle, in one picture

The sweep runs once a day, then every finding splits into one of two lanes
— which lane depends on the **mechanism**, not on how the finding was
found:

```
                          ┌─▶ LANE A (auto-accept)
                          │   risk-only, confidence bar cleared
                          │   ┌────────────────────────────────┐
                          │   │ applied directly: dict entry +   │
                          │   │ CSV patch + audit-log line       │
 1. SWEEP                 │   │ -- live on the board same day    │
 ┌─────────────┐          │   └────────────────────────────────┘
 │ scheduled   │──finding─┤
 │ task runs   │          │
 │ daily,      │          └─▶ LANE B (human review)
 │ searches    │              ANY projection cut, or a risk-only
 │ the news    │              finding below the confidence bar
 └─────────────┘              ┌────────────────────────────────┐
                               │ drafted into pending_news_       │
                               │ adjustments.md -- you say what   │
                               │ to apply, I wire it in           │
                               └────────────────────────────────┘
```

**The rule that decides the lane** (policy §1a): a risk-only bump
(`INJURY_MANUAL_OVERRIDES`) that's well-sourced enough to act on at all
goes straight to Lane A — no one waits on it. A projection-points cut
(`PROJECTION_MANUAL_ADJUSTMENTS`) always goes to Lane B, no matter how
confident the sourcing is, because that lever moves rank/OUR SCORE
directly. A risk-only finding that ISN'T well-sourced enough also lands in
Lane B, not the trash — mechanism only decides who signs off once
something has already cleared the bar for acting on at all.

## The pieces

| Piece | What it is | Lives at |
|---|---|---|
| **The scheduled sweep** | Runs daily, searches for camp/injury/suspension news, filters to relevant players, drafts candidates | `nfl-camp-news-watch` task — file at `C:\Users\19054\.claude\scheduled-tasks\nfl-camp-news-watch\SKILL.md` |
| **The policy** | The rules the sweep (and I) follow: which mechanism to use, how to size an entry, what confidence is enough to act | [`research/news_override_policy.md`](news_override_policy.md) |
| **The pending queue (Lane B)** | Candidates awaiting your decision — nothing here is live | [`research/pending_news_adjustments.md`](pending_news_adjustments.md) |
| **The audit log (Lane A)** | Everything applied WITHOUT your review — the record that makes auto-accept reversible/checkable after the fact | `research/applied_news_overrides_log.md` |
| **Risk-only overrides** | Bumps a player's `injury_score` (1–5 scale) — for real-but-undecided situations. Lane A lives here. | `INJURY_MANUAL_OVERRIDES` in `draftkit/scripts/build_risk_variables.py` |
| **Projection overrides** | Cuts a player's season points by a % — only for real, *decisive* situations (a confirmed return week, a confirmed role split). Lane B only — never auto-applied. | `PROJECTION_MANUAL_ADJUSTMENTS` in `draftkit/draft_analysis.py` |

## Example of each piece, with real content

Not hypotheticals — these are the actual entries in the repo right now.

**1. The scheduled sweep's report-back** (what you get notified with):
> Ran the 4-query sweep. Checked 8 candidate names against the top-150 board and existing overrides. 6 were already covered or out of scope (Pearsall, Kirk, Mahomes, Pierce, Charbonnet, Jeshaun Jones). 3 new: Burden and Nacua drafted as candidates, Love checked and logged as not-actionable. See `research/pending_news_adjustments.md`. Nothing applied to live scoring.

**2. The policy, applied to a real decision** (from `news_override_policy.md`):
> Nacua: a real, unresolved suspension risk, multiple named outlets, but "no discipline decided" (nothing decisive) → policy §1 says **risk-only**, not a projection cut. Severity is a live legal risk with no physical injury attached → policy §2's tier table puts that at **3.5**, same tier as Jacobs' entry.

**3. A pending-queue entry** (from `pending_news_adjustments.md`, awaiting your decision):
```markdown
## Luther Burden -- 2026-08-26
- **What**: Suffered a groin injury at Bears practice (Aug 8), expected to
  miss the rest of the preseason; team "hopeful" for Week 1.
- **Source**: DAZN, CBS Sports camp injury trackers (2026-08-24/25 roundups)
- **Suggested mechanism**: INJURY_MANUAL_OVERRIDES, score 3.5
- **Reasoning**: "Hopeful for Week 1" is short of a decisive timeline, so
  risk-only, no projection cut.
- **Confidence**: Medium-high. Multiple outlets, consistent detail, but no
  single definitive return date yet.
```

**4. A live risk-only override** (already applied, from `INJURY_MANUAL_OVERRIDES`):
```python
"Ashton Jeanty": {
    "score": 4.0,
    "reason": "Sprained ankle at practice (Aug 24) -- not considered long-term "
              "per reporting, but return timeline still unconfirmed",
    "date": "2026-08-26",
},
```

**5. A live projection override** (already applied, from `PROJECTION_MANUAL_ADJUSTMENTS`):
```python
"Jordyn Tyson": {
    "pct": -50.0,
    "note": "Hamstring re-injury -- real, decisive timeline: out until Week 9",
    "source": "User-reported 2026-08-17; underlying injury independently "
              "corroborated via web search (ESPN/NBC Sports/ProFootballRumors)",
    "date": "2026-08-17",
},
```
This one also carries a longer inline comment in the actual file explaining
the -50.0% math (missing ~8 of 17 games, a disclosed rough estimate, not a
precision model) — worth reading in place if you want the full reasoning,
not just the final number.

## How to do the common things

**See what's pending review (Lane B)** → open [`research/pending_news_adjustments.md`](pending_news_adjustments.md). Empty file means a quiet day, not a broken task.

**See what's been auto-applied (Lane A)** → open [`research/applied_news_overrides_log.md`](applied_news_overrides_log.md). This is the one worth spot-checking occasionally, since nothing gated it going in.

**Apply a pending (Lane B) entry** → tell me which one(s). I wire it into the right dict, patch `data/processed/risk_variables.csv` to match (a full rebuild needs a data file this environment doesn't have — see CLAUDE.md), verify it on the live board, then delete the entry from the pending file.

**Undo something auto-applied (Lane A)** → tell me which player looks wrong. Same reversal either way: edit the dict entry (or remove it), re-patch the CSV, log the correction.

**Add something yourself, right now, without waiting for the sweep** → just tell me the news (like Jacobs/Jeanty earlier). Same process, immediate instead of on the daily schedule — and it still follows the Lane A/B split (a risk-only finding I'm confident in, I'll apply directly and tell you; a decisive-enough case for a projection cut I'll still bring to you first).

**Change how often it runs, or what it searches, or the lane split itself** → edit the cron schedule / queries in the task's `SKILL.md`, or the rules in `news_override_policy.md` §1a, or ask me to.

**Check whether the sweep actually ran** → each run notifies you with two parts: what it auto-accepted (Lane A, named explicitly) and what it added to the review queue (Lane B) — never blurred together.

## Current state (as of 2026-08-26)

- **Schedule**: daily, 7:19 AM local.
- **Model**: set to `haiku` in the task's frontmatter — *unconfirmed* whether the scheduler actually honors this field (not a documented option on the scheduling tool). Worth checking after a couple of real runs.
- **Mode**: two-lane, as of today (this was a same-day policy change — see below). Risk-only + high-confidence auto-applies (Lane A); any projection cut always waits for you (Lane B).
- **Scope**: top ~150 players by ADP.
- **Pending in Lane B right now**: Burden (injury, uncovered) and Nacua (suspension risk, uncovered) — both drafted BEFORE the Lane A/B split existed, so they're sitting in the review queue rather than auto-applied. Love was checked and logged as not-actionable.
- **Lane A audit log**: empty so far — the split was just added, nothing has gone through it yet.
- **Already covered** (won't be re-flagged): Jeanty, Jacobs, Tyson, DeVonta Smith, Luther Burden (role-only reason), Corum.

## Known gaps / things worth your critique

- **The Lane A/B split is untested.** It was designed and wired into the policy/task just now, in this same session — no finding has gone through the new branch yet. The first real proof is whichever story the 7:19 AM run (or a manual run) hits next.
- **Auto-accept has no cooling-off period.** A risk-only finding can go from "found" to "live on the board" in the same run, same day, with zero elapsed time for you to catch a bad call before it's already applied. The audit log makes it reversible, not prevented.
- **The confidence bar is still a judgment call, not a hard rule.** "Multiple named outlets" is the standard (policy §3), but there's no mechanical check enforcing it — it still relies on the model reading sources honestly each run.
- **The model-pinning is unverified.** If `haiku` isn't actually being used, that also means Lane A's auto-apply judgment calls are being made by whatever model actually IS running — worth confirming, given Lane A now has real write access.
- **No expiry.** Neither lane automatically revisits an entry once the situation resolves (player activated, suspension decided, or a risk-only entry that should graduate to a projection cut once a timeline firms up). That's still a manual "remember to check."
- **Scope is ADP-only.** A deep-league or bench-relevant story below ADP 150 will never surface, by design — worth confirming that's actually the cutoff you want.
- **One sweep, four fixed queries.** It won't catch something outside those four search shapes.
