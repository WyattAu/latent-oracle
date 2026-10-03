# Research Notes — Hyperparameter Optimization Methodology

**Scope:** how to SEARCH the hyperparameter space systematically instead of
one-at-a-time hunches, sized to our compute (1 GPU + 6 CPU cores, SPRT as
the only trusted objective).

---

## 1. The objective problem

Training loss does not reliably rank engine strength at our scale. SPRT Elo
is the objective but costs 2-3 h per evaluation (200 games). Any HPO scheme
must budget ~1-2 SPRTs per candidate per stage.

### 1.1 Proxy ladder (cheap → trusted)
| Rung | Metric | Cost | Correlation with Elo |
|---|---|---|---|
| 0 | train loss | free | weak |
| 1 | held-out move-matching acc | free-ish | medium (see RESEARCH-EVAL.md) |
| 2 | puzzle-suite accuracy (fixed 500 positions) | minutes | medium |
| 3 | 400-game fast SPRT (15+0.1) | ~15 min CPU | good |
| 4 | 400-game official SPRT (60+6) | 2-3 h | trusted |

HPO optimizes rung 2-3; final candidates graduate to rung 4.

---

## 2. Search strategies

### 2.1 Successive halving / ASHA (recommended core)
- Train N candidates for 10% epochs → keep top half by rung-2 metric →
  retrain survivors to 30% → halve again → final 1-2 to full training.
- Fits GPU budget: 8 candidates × partial epochs ≈ 2 full trainings.
- Use for: LR, weight-decay, GabLR (separate LR for gab_table), decisiveness
  cutoff (150cp), Muon LR pair.

### 2.2 Bayesian (Optuna TPE) around the ASHA winner
- 10-15 trials over the 3-4 most sensitive knobs only (LR, wd, epochs).
- Parallelism: TPE suggests, ASHA prunes — Optuna supports both natively.
- Only after ASHA establishes the basin; else TPE wanders.

### 2.3 Population-based training (PBT)
- Online adaptation (LR perturbation + keep-best exploitation) during ONE
  long training with 4-6 population members. Needs 4-6 concurrent trainings
  → 6GB GPU can't fit 6×6.4M models at batch 512. **Reject at current
  hardware**; revisit if we get a bigger card.

### 2.4 What NOT to search
- Architecture dims (d, layers, heads): data-bound regime; capacity doesn't
  break the ceiling (gold-standard study). Fix at 6.4M.
- Batch size: fixed by GPU memory. 
- Anything with <1 expected Elo effect: SPRT noise floor swamps it.

---

## 3. Concrete HPO plan for v0.3 (budgeted)

Stage A — trainer-side ASHA (GPU, ~1 day total):
1. Candidates (8): LR ∈ {1.5e-4, 3e-4, 6e-4} × {cosine, const} × {wd 0.01, 0.1},
   plus Muon pair (0.02/6e-4, 0.05/3e-4).
2. Each: 0.3 epoch on bc_v1_combined.shard subset (5M records), rank by
   held-out move-matching.
3. Survivors (3): 1 full epoch, rank by puzzle suite.
4. Winner: full 3-epoch train → official SPRT.

Stage B — inference-side grid (CPU SPRT, ~1 day):
1. RecyclePasses ∈ {1, 2, 4} × LoopCDAlpha ∈ {0, 0.25, 0.5, 1.0} on the
   winner net — 11 configs, but R=1/α=0 is the baseline → 10 × fast SPRT
   (15+0.1, 400 games ≈ 15 min each ≈ 2.5 h).
2. Top-2 configs → official 60+6 SPRT.

Stage C — fine-tune chain knobs (after labeled_5m lands):
1. Decisiveness cutoff ∈ {100, 150, 250}cp via ASHA-lite (3 candidates,
   0.2-epoch each).
2. Policy-value loss ratio ∈ {0.5, 1.0, 2.0}.

## 4. Infrastructure

- `optuna` + `torch` only; ASHA implemented as a ~80-line scheduler script
  (no heavy framework needed at our scale).
- Each trial writes `runs/hpo/<id>/` with config.json + eval.json so the
  index is reproducible.
- Guardrail: every trial seeds identically (seed 0) and evals on the same
  50k-record held-out slice → paired comparisons, lower variance.

## 5. Synergies with current work

- EMA weights (RESEARCH-OPTIMIZATION2.md §1): search EMA decay {0.999,
  0.9995} in stage A tail — 2 extra candidates, export both, SPRT.
- Decisiveness weighting (H2): if H2 SPRT (running) is positive, its cutoff
  becomes stage-C search center.
- Mirror augmentation (RESEARCH-DATA.md §1): A/B once (on/off) rather than a
  search axis — expected-positive everywhere.
