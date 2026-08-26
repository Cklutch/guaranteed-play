# Pending news adjustments -- human review queue

Drafted by the `nfl-camp-news-watch` scheduled sweep (or a manual run of the
same process). **Nothing here is applied to the live scoring** -- these are
candidates only. To apply one, wire it into `INJURY_MANUAL_OVERRIDES`
(`draftkit/scripts/build_risk_variables.py`) and/or
`PROJECTION_MANUAL_ADJUSTMENTS` (`draftkit/draft_analysis.py`), matching the
dated/sourced/reasoned format already established there, then re-run/patch
`data/processed/risk_variables.csv` the same way prior entries were applied.
Once applied, delete the entry from this file.

---

## Luther Burden -- 2026-08-26
- **What**: Suffered a groin injury at Bears practice (Aug 8), expected to miss the rest of the preseason; team "hopeful" for Week 1.
- **Source**: DAZN, CBS Sports camp injury trackers (2026-08-24/25 roundups)
- **Suggested mechanism**: `INJURY_MANUAL_OVERRIDES`, score 3.5 -- his current `injury_score` is still 1.0 (untouched), even though he's missing real practice time right now. Note: he already has an unrelated `PROJECTION_MANUAL_ADJUSTMENTS` entry (+10%, DJ Moore trade/camp-role buzz) -- that one is about opportunity, not this injury, so it stays; this would be a separate, additional injury-score entry, not a replacement.
- **Reasoning**: "Hopeful for Week 1" is short of a decisive missed-games timeline (same standard the codebase already applies -- see Jeanty's 2026-08-26 entry), so risk-only, no projection cut.
- **Confidence**: Medium-high. Multiple outlets, consistent detail (Aug 8 injury, preseason-out framing), but no single definitive return date yet.

## Puka Nacua -- 2026-08-26
- **What**: A real, still-unresolved NFL suspension risk tied to a civil lawsuit (alleged biting + antisemitic remark); Schefter has said missing Week 1 vs. SF "can't be ruled out." As of this week, described by SI/Yahoo as genuinely "a coin flip" -- no discipline decided, no charges filed as of this entry.
- **Source**: NBC Sports (Schefter, 2026-08-11), SI.com Rams beat (multiple pieces through this week), Yahoo Sports
- **Suggested mechanism**: `INJURY_MANUAL_OVERRIDES`, score ~3.5 (same tier/reasoning as the existing Josh Jacobs entry, 2026-08-26 -- a real, live, undecided legal/suspension risk, not a physical injury). His current `injury_score` is 1.4 -- essentially reflects none of this.
- **Reasoning**: Nothing is decided, so no `PROJECTION_MANUAL_ADJUSTMENTS` discount -- same standard as Jacobs. This is meaningfully higher-stakes than most entries here given his ADP (4.0, a top-4 overall pick) -- worth prioritizing for review even though the risk-only treatment is the same shape as Jacobs.
- **Confidence**: High that the situation itself is real and unresolved (many independent, named-source outlets); low on which way it resolves (sources themselves call it a coin flip).

## Jeremiyah Love -- 2026-08-26 (logged, not actionable)
- **What**: Cardinals' first unofficial depth chart lists Tyler Allgeier as RB1, Love (the No. 3 overall pick) as RB2.
- **Source**: ProFootballTalk/NBC Sports, Yahoo Sports, ClutchPoints -- independently corroborated across multiple named outlets.
- **Suggested mechanism**: None right now. Every outlet checked frames this as standard "make the rookie earn it" camp positioning, not a real role threat -- one piece stated explicitly there's "almost no chance" Love isn't Arizona's lead back for the season barring injury.
- **Reasoning**: Logged for visibility only, per instructions to record real-but-not-decisive findings honestly rather than invent a number. A first search summary attached this to a coaching change ("new coach Mike LaFleur") that a second, targeted search could NOT corroborate anywhere -- dropped as a likely inaccuracy rather than included.
- **Confidence**: High on the depth-chart fact itself; not actionable as a scoring change.
