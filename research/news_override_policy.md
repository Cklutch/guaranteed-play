# Manual news-override policy

Governs both `INJURY_MANUAL_OVERRIDES` (`draftkit/scripts/build_risk_variables.py`)
and `PROJECTION_MANUAL_ADJUSTMENTS` (`draftkit/draft_analysis.py`), and the
`nfl-camp-news-watch` scheduled sweep that feeds `research/pending_news_
adjustments.md`. Written 2026-08-26 by extracting the rules already applied,
consistently, across every real entry made that day (Jeanty, Jacobs, Burden,
Nacua, Tyson, Smith, Burden-role, Corum) plus the Love finding that was
correctly logged but NOT acted on. Not a new set of rules -- a write-up of
the ones already in use, so future runs (a scheduled sweep, a different
session, a different person) apply them the same way instead of re-deriving
judgment calls from scratch each time.

## 1. Which mechanism: risk-only, or a projection cut?

**Default to risk-only** (`INJURY_MANUAL_OVERRIDES`, moves `injury_score`
only). Only add a `PROJECTION_MANUAL_ADJUSTMENTS` season-points discount
when there is a real, DECISIVE, specific detail to build one from -- a
confirmed return week, a confirmed depth-chart change with a stated split,
a confirmed trade. "Not considered long-term," "hopeful for Week 1,"
"pending review," "could miss Week 1" -- none of these clear that bar.

This has been the deciding line in every case so far:
- **Tyson** (out until Week 9, a real confirmed date) -> got a -50.0% cut.
- **Jeanty** (sprained ankle, "not long-term" but no confirmed date),
  **Jacobs** (existing injury + pending suspension, nothing decided),
  **Burden's injury** (hopeful for Week 1, no confirmed date), **Nacua**
  (suspension possible, nothing decided) -> all risk-only, no cut.

## 1a. Auto-accept vs. review (added 2026-08-26, user-directed)

The mechanism split in §1 now also decides who signs off:

- **Risk-only entries (`INJURY_MANUAL_OVERRIDES`) that clear §3's confidence
  bar are auto-accepted** -- applied directly (dict entry + CSV patch, see
  §7), no human review gate. This is the lever §1 already calls "cheap to
  be wrong about."
- **Any projection cut (`PROJECTION_MANUAL_ADJUSTMENTS`) always goes to
  human review**, regardless of confidence -- drafted into `research/
  pending_news_adjustments.md`, same as before. This is the lever that
  "moves rank/tier and OUR SCORE directly," per §1's own rationale for
  reserving it for decisive information.
- A risk-only finding that does NOT clear §3's confidence bar still goes to
  review, same as before -- mechanism type only fast-tracks entries that
  were already going to be acted on; it never lowers the bar for acting at
  all.

So: three outcomes per finding, not two -- auto-accepted (risk-only,
high-confidence), drafted for review (a projection cut, OR a risk-only
finding below the confidence bar), or discarded (already covered, out of
scope, or too speculative to act on at all).

Rationale, stated plainly: a wrong injury_score bump costs a few risk
points on the board. A wrong projection cut moves rank/tier and OUR SCORE
directly. Reserve the higher-consequence lever for when the underlying
fact is actually decisive, not just real.

**When a risk-only entry later resolves**: if the timeline firms up to a
real decisive number (a suspension length is announced, a return date is
confirmed), add the corresponding `PROJECTION_MANUAL_ADJUSTMENTS` entry
at that point -- don't wait for a full pipeline rebuild. If the player
instead fully clears (activated, suspension avoided), remove the
`INJURY_MANUAL_OVERRIDES` entry -- it should not linger past the situation
that justified it.

## 2. Sizing the injury_score (1.0-5.0 scale)

Observed tiers, in order of severity -- use the nearest precedent, not a
fresh guess each time:

| Score | When | Precedent |
|---|---|---|
| 5.0 (ceiling) | A CONFIRMED decisive multi-week absence | Tyson, out until Week 9 |
| 4.0 | A real, freshly diagnosed current injury; meaningfully elevated near-term risk, timeline genuinely unconfirmed | Jeanty, sprained ankle Aug 24 |
| 3.5 | Either (a) two independent real-but-uncertain risks compounding, (b) a real, live, undecided LEGAL/suspension risk with no physical injury attached, or (c) an ongoing minor injury with a stated hope of return (softer than a fresh diagnosis) | Jacobs (injury + suspension review); Nacua (pure suspension uncertainty); Burden (hopeful-for-Week-1 framing) |
| Below 3.5 | Not yet used -- reserve for something real but genuinely low-stakes (e.g. a minor, likely-resolved-by-Week-1 tweak) rather than inventing a precise number for it | -- |

These are disclosed judgment calls, not a precision model -- same standard
the codebase already states for `PROJECTION_MANUAL_ADJUSTMENTS` percentages.
Pick the nearest-fitting tier and say why in the note, don't split hairs
between e.g. 3.5 and 3.7.

## 3. Confidence bar to act at all

Act (log, and apply if it clears \S1's bar) when EITHER:
- Multiple independent named outlets report the same core fact, or
- One authoritative insider (Schefter/Rapoport-tier) is corroborated by a
  team/GM statement or a second outlet.

A single, unverified DETAIL inside an otherwise-corroborated story should
be dropped rather than repeated if a second, targeted check can't confirm
it -- don't let one hallucination-shaped claim ride along with a real
story. (Precedent: the Jeremiyah Love depth-chart fact was independently
verified across three named outlets and kept; the attached "new coach
Mike LaFleur" detail could not be corroborated by a follow-up search and
was dropped from the writeup entirely.)

**Uncertain does not mean skip.** A genuinely unresolved, high-stakes
situation (Nacua: sources call it a literal coin flip) still gets logged
and given a risk-only entry -- uncertainty changes which mechanism you use
(\S1), not whether you act.

## 4. Scope filter

Only players within roughly the **top 150 by ADP** are worth flagging --
this app only cares about fantasy-relevant players. Discard news about
anyone deeper (e.g. Jeshaun Jones' suspension: real, but no real ADP,
not board-relevant).

## 5. Don't re-flag what's already covered

Before drafting anything, check ALL of:
- Both override dicts (`INJURY_MANUAL_OVERRIDES`, `PROJECTION_MANUAL_
  ADJUSTMENTS`) for an existing entry on that player.
- The player's CURRENT `injury_status`/`injury_score` in `data/processed/
  risk_variables.csv` -- if it already plausibly reflects the news (e.g.
  already shows IR/PUP with an already-elevated score matching the
  report), the automated Sleeper-status pipeline already caught it; skip.
  (Precedent: Pearsall, Christian Kirk, Mahomes, Alec Pierce, Charbonnet
  were all checked and skipped this way in the same sweep that surfaced
  Burden and Nacua.)
- Prior entries in `research/pending_news_adjustments.md` for the same
  player/story, so a daily sweep doesn't duplicate an item still awaiting
  review.

An existing entry for a DIFFERENT reason doesn't block a new one -- Burden
already had a `PROJECTION_MANUAL_ADJUSTMENTS` entry for role/opportunity
(DJ Moore trade); that didn't stop a separate `INJURY_MANUAL_OVERRIDES`
entry for his unrelated groin injury.

## 6. Documentation standard

Every entry, in either dict or in the pending-review file, states: what
happened, the real source(s), the suggested mechanism and magnitude, the
reasoning for that specific magnitude, and an honest confidence read. This
is not optional formatting -- it's what lets a future reader (human or AI)
judge whether the entry still holds up, rather than trusting a bare number.

## 7. How an auto-accept actually gets applied (added 2026-08-26)

Same mechanics used for every manual entry so far (Jacobs, Jeanty, Burden,
Nacua) -- an auto-accept is not a different, lighter-weight path, just one
with no review gate in front of it:

1. Add the entry to `INJURY_MANUAL_OVERRIDES` in `draftkit/scripts/
   build_risk_variables.py`, full documentation per §6, PLUS a note that
   this was auto-applied: `"applied_by": "nfl-camp-news-watch auto-accept, YYYY-MM-DD"`.
2. Patch `data/processed/risk_variables.csv` directly: set the row's
   `injury_score` and `injury_override_note`, then recompute `risk_index`
   by calling `draftkit.risk_scoring.player_from_variable_row()` and
   `risk_index()` on that row with `load_weights()` -- the SAME functions
   the real pipeline uses, so the patch is numerically identical to what a
   full rebuild would produce (a full rebuild isn't runnable in this
   environment -- see CLAUDE.md's Gotchas -- so the code edit is the
   durable source of truth for next time one runs; the CSV patch is what
   makes it live today).
3. Append a line to `research/applied_news_overrides_log.md` (create if
   missing) -- player, date, score, one-line reason, "auto-accepted."
   This is the audit trail: a running record of everything the system
   changed without a human clicking accept, so it can be reviewed and, if
   wrong, reverted (edit the dict entry, re-patch the CSV) even though no
   one approved it going in.
4. The daily report-back names anything auto-accepted explicitly, not just
   the review-queue additions -- "silently correct" isn't good enough for
   something that changed the live board unsupervised.
