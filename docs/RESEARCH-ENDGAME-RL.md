# Research Notes — Endgame Ground Truth, Gumbel RL, Trajectory Consistency

**Scope:** round 8. Anchored on a measured weakness: our puzzle-suite
endgame bucket is the worst (45% vs 54% decisive) and endgame conversion is
where searchless engines bleed half-points. Three techniques, all cheap,
all grounded in existing infrastructure.

---

## E1. Syzygy ground-truth training (ADOPT — highest ROI in the portfolio)

### The insight
We already OWN perfect endgame labels: 290 Syzygy 3-4-5-man files (WDL +
DTZ) and vendored Fathom, probing in µs. Our shard's ≤5-man positions can
be rescored with EXACT game-theoretic values — no SF, no depth error, no
decisiveness heuristics. ~6-9% of a Lichess shard reaches ≤5 pieces; for
50M records that's ~3-4M perfectly-labeled endgame positions, free.

### Design
1. Sweep the combined shard; for each record with ≤5 pieces, probe WDL
   (win/draw/loss) and DTZ (distance to zeroing).
   MEASURED coverage: ~0.57% of shard records are ≤5 pieces (170/30k on
   labeled_1m; games rarely reach 5 pieces) → 50M records yield ~280k
   exact-label positions. Still worth it — perfect labels exactly where
   the model is weakest — but it is a precision patch, not a data flood.
2. Value target: replace the logistic-eval soft WDL with the EXACT WDL.
3. Policy target: the DTZ-optimal move (probe each legal child, pick
   min-|DTZ| toward the win / max toward the draw) — a *perfect* policy
   label in every covered position.
4. Weight: 2-3× standard loss weight (endgame data is our minority class;
   RESEARCH-DATA §2.1 game-phase balancing composes).
5. Later: extend to 6-man files (DTZ tables ~1-2 TB total — defer; 5-man
   covers the conversion problem; 6-man adds precision, not coverage).

### Expected effect
- Endgame puzzle bucket: 45% → 60%+ plausible (the labels stop being
  approximations exactly where approximation hurts most).
- Conversion of winning positions: the DTZ-optimal policy label teaches
  *efficient* wins, not just winning — fewer 50-move/repertoire stumbles.
- The value head learns exact game theory in the region where the
  logistic-eval approximation (k=0.0037) is worst (long DTM/DTZ lines).

### Implementation map
- Fathom probe already wrapped (`tb::probe_root` engine-side, tested).
- Trainer-side: add a small `syzygy.py` (python-fathom or subprocess to a
  tiny probe binary) OR — cheaper — do the rescoring in the RUST pipeline
  (`lo-data` already links Fathom? no — engine-side only). Pragmatic path:
  a one-shot python script probing labeled_*.shard positions, writing a
  `tb_flags` sidecar (2 bits WDL + 1 bit has-tb), consumed by the trainer.
- Composes with: two-hot eval targets, decisive weighting, aux heads.

### Verdict: ADOPT in the AV phase. Cost: ~1 day (script + trainer flag).

---

## E2. Gumbel-GRPO — policy-improvement-guaranteed sampling for the RL stage

### Background
"Policy improvement by planning with Gumbel" (Danihelka et al., ICLR 2022;
validated at scale by Gumbel AlphaZero/MuZero in MiniZero 2023): sampling
actions with the Gumbel-top-k trick turns even K=2..16 rollouts into a
*provably improved* policy (consolidated sigma-weighted visit distribution)
— the guarantee plain temperature sampling lacks.

### Application to our GRPO design (RESEARCH-RL §3)
- Sample the K=16 group moves via **Gumbel-top-K from the policy prior**
  (replaces temperature sampling): `a_i = GumbelTopK(softmax(logits))`.
- After scoring, build the consolidated target: mix the policy prior with
  the group's score-ranked counts (the paper's sigma-weighted improved
  policy) and add a KL term toward it — the RL update then satisfies the
  improvement bound instead of merely following the empirical gradient.
- Practical win: at K=16 with high-entropy chess policies, plain GRPO's
  group baseline is noisy; Gumbel's guaranteed-improvement target reduces
  variance and prevents the "all K samples are the same move" degenerate
  groups at low temperature.

### Verdict: ADOPT as the RL-stage sampling rule (design change, ~30 lines).

---

## E3. Trajectory-consistency loss on shard pairs (PILOT)

### Idea
The shard's consecutive records form game continuations (the diffusion
pilot's bridging already mines them). A value-consistency loss:
```
V(s_t) ≈ γ · V(s_{t+1}) + (1-γ) · result      (γ ≈ 0.95)
```
(TD(0)-style Bellman residual on static human games — Bellman residual
minimization is classical; the novelty angle is that our pair-bridging
makes it free, and it regularizes the value head toward *temporally
coherent* evaluations, which the independent-position training never
enforces.)

### Risks
Bellman residuals are notoriously unstable at weight 1 — use weight ≤0.1,
γ fixed, pairs filtered to |eval delta| sanity (skip captures of queens
changing eval by 500cp... no — those are LEGITIMATE deltas; skip only
non-consecutive bridges, which the bridging already handles).

### Verdict: PILOT alongside the AV phase (loss term, flag-gated).

---

## E4. Endgame conversion metrics (eval-suite addition)

Two numbers to track per SPRT game batch (script over PGNs, no engine
changes):
1. **Conversion rate**: among games where the net had a ≥+2 eval at some
   point, fraction won.
2. **Replay/flag rate**: average repetitions per game, time-forfeit counts.
These become the mechanism checks for E1 (and N1 HiCo later). Baseline the
BC-v1 chain SPRTs first (PGNs already archived) so the improvement claim is
paired.

---

## Summary table

| Item | Cost | Expected | Verdict |
|---|---|---|---|
| E1 Syzygy ground-truth endgame labels | ~1 day | endgame bucket +15pts, clean conversion | **ADOPT (AV phase)** |
| E2 Gumbel-GRPO sampling | ~30 lines | lower-variance RL, improvement bound | **ADOPT (RL stage)** |
| E3 TD-consistency on shard pairs | ~20 lines | value coherence | PILOT |
| E4 conversion/replay metrics | ~50 lines script | mechanism evidence | ADOPT (next SPRT batch) |
