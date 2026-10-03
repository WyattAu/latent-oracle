# Research Notes — Training

**Scope:** training-target, data, and optimization findings for the 6.4M
searchless-chess model. Companion to RESEARCH-ARCHITECTURES.md and
RESEARCH-HYPERPARAMS.md.

---

## 1. Gold-standard lightweight agent study (Jul 2026)

**Paper:** "A Gold-Standard Study of What Makes a Lightweight Game-Playing
Agent Strong" — 100+ controlled runs against a fixed rule-based expert.

### 1.1 What helped (stacked: ~30% → ~36% win rate)
1. **Trust-region-style updates** — stable policy improvement.
2. **Well-aimed reward** — reward pointed at the actual objective.
3. **Curriculum of tougher opponents** — opponents graduated in strength.
4. **Warm starting** — initialize from a prior trained model.
5. **Keeping the best checkpoint** — evaluate-and-retain, never regress.

### 1.2 What did NOT help
- Reward shaping (short- and long-term).
- Learned state embeddings.
- **Imitation learning / DAgger** — unhelpful at lightweight scale. *This is a
  third-party replication of our H1 result (SF-label imitation hurt us).*
- Live LLM opponents.
- **Extra capacity** — "extra capacity does little to break the ceiling." The
  ceiling is set by data/feedback quality, matching our finding that BC-v0
  (2000-Elo human data) caps around 2000-Elo play.

### 1.3 Application to our pipeline
- Warm starting: already in the plan (BC pre-train → AV fine-tune → quality
  fine-tune chain in SPEC-OPTIMAL-MODEL.md §4). Validated.
- Keep-best: our trainer exports per-epoch; add SPRT gating so the "best"
  blob is chosen by measured Elo, not last-epoch. TODO in phase scripts.
- Curriculum: for AV fine-tuning, order shards by eval magnitude
  (equal positions first → decisive positions last), or simpler: the
  decisiveness weighting below achieves the same reordering implicitly.
- Anti-goal confirmed: do NOT chase a bigger model before exhausting data
  quality levers.

---

## 2. Decisiveness weighting (implemented 2026-10-03)

**Motivation:** H1 showed SF-label distillation with uniform cross-entropy
degrades searchless play by ~309 Elo vs BC. Diagnosis: in equal positions the
"best" move is near-arbitrary, so hard CE on it is label noise. Lc0's policy
training has used the same insight (weight policy loss by position
decisiveness).

**Implementation:** `--decisive-weighting` in trainer/train.py (commit
`3ce2f02`): labeled records get policy weight `clamp(|eval|/150, ≤1)`; BC
records keep weight 1.

**Test queued (H2):** retrain on identical labeled_1m.shard with weighting,
SPRT vs the −436 uniform baseline. Direct mechanism comparison.

---

## 3. Muon optimizer family

**Papers:** "Adam Improves Muon: Adaptive Moment Estimation with
Orthogonalized Momentum" (Feb 2026), "Nonsmooth Optimization via Orthogonalized
Momentum" (Sep 2026), AF-Muon (Oct 2026, tied-embedding variant).

### 3.1 What it is
Muon orthogonalizes each weight matrix's momentum (Newton–Schulz
approximation) before applying the update — the update direction is
whitened in matrix space. Consistently faster convergence than AdamW for
hidden-layer matrices in LLM training at equal compute.

### 3.2 Practical recipe for us (6.4M params)
- Muon on all 2-D hidden weights (attention projections, FFN matrices,
  policy embeddings), LR ≈ 0.02–0.05, momentum 0.95, Nesterov on,
  no weight decay.
- AdamW on embeddings and the value head (1-D or tied params), LR 3e-4–6e-4,
  weight decay 0.1. AF-Muon covers the tied-embedding case if we tie.
- Newton–Schulz runs on GPU; overhead at our matrix sizes (256×1024) is
  negligible (<2% step time).

### 3.3 Expected gain
Speed more than final quality at small scale, but at fixed wall-clock
(RTX 2060) faster convergence ≈ more effective epochs. Estimate +10–30 Elo
or ~20% training-time reduction.

---

## 4. Distillation refinements

### 4.1 Temperature
For AV targets: convert SF eval cp → move probabilities with a temperature.
`p_i ∝ exp(eval_i / τ)`. With only top-3 moves labeled, distribute remaining
probability mass over the other legal moves uniformly scaled down. τ ≈ 150cp
matches the logistic k=0.00368208 already used in our value loss (1/k ≈ 272 —
logit scale differs; keep the logistic form for value, softmax-τ for policy).

### 4.2 What NOT to distill (H1 lessons)
- Uniform hard targets from a single engine depth: rejected.
- Depth 16 labels at 6.4M params: the model cannot represent the precision
  gap; depth 14 labels + weighting is the pilot (5M shard in progress).

### 4.3 Multi-PV soft targets (format change, deferred)
True per-move AV needs per-move evals in the shard (currently only best-move
eval + top-3 moves). Candidate format v2: append 3×i16 per-move evals.
Cost: shard format migration. Do only if decisiveness weighting alone
under-delivers in H2.

---

## 5. Curriculum learning

**Evidence:** the gold-standard study (§1) validated opponent-strength
curricula; for supervised searchless training the analog is **position
difficulty ordering** — proxy = |eval| (decisiveness) or game phase.

Options ranked by implementation cost:
1. Decisiveness weighting (implemented) — implicit, no data reordering.
2. Two-stage fine-tune: epochs on equal-heavy subset, then decisive-heavy
   subset. Trivial shard split by |eval|.
3. Full curriculum scheduler. Overkill.

---

## 6. Scaling-law sanity check for our regime

- Chinchilla-style optimal data ≈ 20 tokens/param → 6.4M params "wants"
  ~130M positions/epoch. We have 50M → ~2.6 epochs is the compute-optimal
  point; our 3-epoch default is close.
- The 2026 small-scale-experiments literature ("Small-Scale Experiments: Are
  We There Yet?") supports using small models to predict large-model
  rankings for data-target choices, which is exactly our methodology.
- Model-scaling honest assessment: with 50M positions, going 6.4M → 14M
  params is data-undernourished. The gold-standard study agrees (capacity
  doesn't break ceilings). Keep 6.4M until the 5M-labeled AV phase proves
  data-bound, not capacity-bound.
