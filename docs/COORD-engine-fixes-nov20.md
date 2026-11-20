# Coordination note — engine fixes branch (2026-11-20)

This branch carries the engine/search session's work, pushed here because
main was mid-rebase (another session's 84-commit RESULTS rewrite).

Contains:
- fix(engine) x4: startpos-moves handler, ~uint8_t Color, is_draw 3-fold,
  ABNodes parse (the AB known-bug TODO's real root causes)
- feat(nn): evaluate_value value-only parallel path + nnbench
- docs: RESULTS.md entries (sign audit, ~1940 Elo calibration, the search
  arc, the value-head human-noise diagnosis) + ROADMAP dual-track section

Note for the RESULTS-rewrite session: your interactive rebase was aborted
after my stash conflicted with it; your commit series is intact in this
branch's history (it was an ancestor of the pre-abort HEAD). The two
doc series can be merged textually — the engine entries do not overlap
the d16 research-loop entries.
