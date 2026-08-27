# research/ -- index

Four docs, one system: keeping the live board current with real, disclosed
news the data pipeline can't see until its next (infrequent) rebuild. Start
here after time away rather than re-deriving which file does what.

- **[`NEWS_OVERRIDES_README.md`](NEWS_OVERRIDES_README.md)** -- start here.
  The news-override system explained end to end: the two-lane (auto-accept
  vs. human-review) lifecycle, real worked examples, and where every piece
  lives.
- **[`news_override_policy.md`](news_override_policy.md)** -- the actual
  rules: which mechanism to use (risk-only vs. a projection cut), how to
  size an `injury_score`, what confidence bar justifies acting, and how a
  finding actually gets applied.
- **[`pending_news_adjustments.md`](pending_news_adjustments.md)** -- a
  historical log of findings that were checked but never needed a queue
  entry (kept so a future sweep doesn't re-flag them); the live review
  queue itself moved to `pending_news_adjustments.json`, resolved via the
  in-app News Queue page (`pages/2_News_Queue.py`).
- **[`MODEL_REGISTRY.md`](MODEL_REGISTRY.md)** -- what's eligible for live
  scoring vs. research-only; check this before wiring any research model
  into rankings, tiers, or recommendations.

Also in this directory: [`archetype_engine_design.md`](archetype_engine_design.md)
-- design notes for the QB/RB/WR/TE usage-archetype feature engineering,
not yet part of this index's system.
