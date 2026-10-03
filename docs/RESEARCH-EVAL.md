# Research Notes — Evaluation Methodology

**Scope:** making the numbers trustworthy: SPRT power, Elo confidence
intervals, proxy metrics, and the promotion policy between them.

---

## 1. SPRT at our game counts

### 1.1 Current setup
- fastchess SPRT, bounds [-35, +35], α=β=0.05 default, 400 games
  (2×200) at 60+6, openings.epd random, 5 concurrency.
- Reported: Elo ± CI.

### 1.2 Statistical reality check
- Standard error of a 400-game match ≈ 2000/√400 ≈ 100 Elo *unpaired*;
  with openings reuse + color pairing the effective SE ≈ 30-50 Elo.
- Our historical CIs (±33, ±37, ±43, ±46) match this. **Differences below
  ~30 Elo are not resolvable at 400 games.**
- Power for detecting +30 Elo at H0=0, H1=+35: ~50% (coin flip!). Power
  hits 80% only at ~1200 games. Implication: use 400-game SPRT for
  *large* effects (H1-style verdicts) and 1200-game confirmations for
  close calls (±30-50 differences, e.g., H2 vs BC gap).

### 1.3 Practical policy
| Effect size (measured) | Verdict | Action |
|---|---|---|
| CI excludes 0 by >30 | strong | accept/reject, log |
| CI includes 0 | weak | extend to 1200 games IF decision-relevant |
| |Elo| < 15 | noise | treat as tie, prefer cheaper/simpler variant |

---

## 2. Fast triage: 15+0.1 blitz SPRT

The gold-standard study and Lc0 practice both use fast triage:
- 15+0.1, 400 games, 5 concurrency ≈ 12-18 min per candidate.
- Elo estimates are compressed vs 60+6 (smaller advantage of strength) but
  rank order holds for effects > 50 Elo.
- Use for: HPO stage B grids (10+ candidates), LoopCD α sweeps.
- Danger: PST fallback triggers at <10s remaining → low-clock behavior
  pollutes. For triage, set `tc=15+0.2` and verify time-forfeit rate <2%
  per side in the PGNs.

---

## 3. Proxy metrics (no games needed)

### 3.1 Move-matching accuracy
argmax-policy == human/SF-move rate on a fixed held-out slice (50k records).
- Cheap: one forward pass batch.
- Known caveat (2026 "Tracing the Thought"): move-matching saturates while
  strength keeps improving; treat as a *ranking* signal with monotonicity
  assumption only within one training family, not across architectures.
- Our usage: HPO stage A ranking only (paired, same slice).

### 3.2 Puzzle suite
500 fixed positions with SF-d20 "solution" move, stratified: 200 quiet
(|eval|<50), 200 decisive (|eval|>200), 100 endgames (≤12 pieces).
- Closer to play strength than move-matching (picks THE strong move, not
  THE human move).
- Build once from labeled_5m.shard (script `tools/make_puzzles.py`, TODO).

### 3.3 Value calibration
Reliability of WDL head: bucket predicted pwin, measure realized score on
the held-out slice. Feeds the engine's draw gates (engine.cpp pwin>0.6
threshold is currently arbitrary).

---

## 4. Promotion policy (what gets deployed)

1. Every new net → 400-game official SPRT vs SF16-p1 (and p2 for verdicts).
2. Winner of HPO grids → confirm at 1200 games vs the *current best net*
   (head-to-head, not just vs SF pool) before it becomes the new baseline.
3. All results logged to `data/sprt/` with PGNs retained (we do this).
4. Keep-best gate (`keep_best.sh`) enforces rule 2 automatically across
   epoch exports.

## 5. Known biases to watch

| Bias | Source | Mitigation |
|---|---|---|
| Opening book leakage | 2000 fixed EPD reused | regenerate every ~5k games from shard |
| Time-forfeit asymmetry | nets differ in per-move cost | check forfeit counts in every txt |
| Contempt/draw rate drift | value head calibration | §3.3 calibration check each generation |
| CPU contention | labeling running during SPRT | keep_best.sh waits for labeler (implemented) |
