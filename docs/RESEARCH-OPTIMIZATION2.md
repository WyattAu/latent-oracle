# Research Notes — Optimization Round 2: Weight Averaging, Loss Refinements, Self-Play FT

**Scope:** second-pass optimization research — techniques that improve a *trained*
model without more data or parameters. Complements RESEARCH-TRAINING.md.

---

## 1. Weight averaging (EMA / SWA) — cheapest known win

**Evidence:** Lc0 trains every network with an **EMA (exponential moving
average) of weights** used as the playing net; SWA/EMA literature shows flat
minima generalize better across domains. The 2026 model-merging literature
(energy-proportional rank allocation, SAM-on-the-curve) confirms weight-space
interpolation between checkpoints stays in flat regions.

### 1.1 Recipe
- Maintain shadow params: `ema_w ← 0.999 · ema_w + 0.001 · w` every step.
- Export BOTH `net_eN.bin` (raw) and `net_eN_ema.bin`.
- SPRT decides which plays stronger (never assume).

### 1.2 Cost / expected gain
- Cost: one extra param buffer (~26MB GPU for 6.4M fp32) + one fused op/step.
- Expected: +20–50 Elo equivalent (fewer training-noise spikes), and EMA nets
  are consistently more robust under quantization (relevant to our INT8 work).
- Verdict: **ADOPT** — implement in trainer, export both variants.

---

## 2. Loss refinements for the policy head

### 2.1 Label smoothing vs decisiveness weighting
Label smoothing (ε≈0.1) uniformly softens targets; our decisiveness weighting
(is implemented) targets the *positions* where the label is unreliable.
They compose: smoothing on top of weighting risks over-softening the moves
that matter. Verdict: keep weighting only; **reject** ε-smoothing for policy.

### 2.2 Policy goal error (Lc0 metric)
Lc0 monitors *policy goal error*: cross-entropy of the policy restricted to
the move the search would have played (or the played move in supervised
mode), tracked separately from value loss. We log policy+value already;
adding a **move-matching accuracy** line to the eval loop (argmax policy ==
label, on a held-out slice) gives a strength-correlated metric that doesn't
need SPRT. Verdict: **ADOPT** (cheap logging, better train/eval signal).

### 2.3 z-loss (already recommended in RESEARCH-HYPERPARAMS.md)
Logit-drift guard for long fine-tune chains; keep 1e-3.

---

## 3. Self-play fine-tuning (SPIN) for a supervised engine

**Evidence:** SPIN and descendants (FormulaSPIN 2026, IRIS, "Triplets Better
Than Pairs" 2026) show iterative self-play fine-tuning improves LLMs beyond
their supervised data by discriminating model outputs from expert data.

### 3.1 Engine adaptation
1. Play N fast self-play games (no search, pure policy + value gates).
2. Keep positions where the **value head** and the played move agree with
   (a) SF label or (b) game outcome — self-distillation with a quality filter.
3. Fine-tune on the curated self-play positions mixed with original BC data
   (1:3 ratio to avoid drift).

### 3.2 Honest assessment
- Our model is 2000-Elo; self-play data is *worse* than the Lichess 2000+
  human data we already train on. SPIN helps when model data approaches
  expert data quality; ours doesn't yet.
- BUT: as a **final polish after the Lc0 fine-tune** (3600-Elo exposure),
  a short 1-iteration SPIN with heavy quality filtering is a plausible
  +10–30 Elo. Verdict: **PILOT (P3)** after AV + Lc0 fine-tune chain lands.

---

## 4. Sharpness-aware minimization (SAM)

Flat-minima optimization; 2× step cost (two forward/backward passes).
2026 work continues validating it. At our batch size on a 6GB card the extra
pass costs real wall-clock; EMA (§1) captures much of the flatness benefit
for free. Verdict: **REJECT for now** — revisit only if EMA under-delivers.

---

## 5. Quantization-aware training (QAT)

Directly relevant to the INT8 NetQ parity project: instead of post-training
quantization, train with fake-quantized weights/activations so the network
adapts. Given INT8 is our deployment path on the VNNI kernel:
- Straight-through estimator on weight quantization during the last
  fine-tune epoch is ~20 lines.
- Expected: recovers most of the INT8 drop automatically.
- Verdict: **ADOPT at the M2 stage** (after INT8 parity bug is bisected).
  Note: QAT works best FROM the EMA net (§1 synergy).

---

## 6. Verdict table

| Technique | Cost | Expected | Verdict |
|---|---|---|---|
| EMA weights + dual export | ~15 lines | +20–50 Elo, robustness | **ADOPT now** |
| Move-matching eval logging | ~10 lines | better signal | **ADOPT now** |
| QAT for INT8 | ~20 lines | INT8 drop recovery | **ADOPT at M2** |
| SPIN self-play polish | pipeline work | +10–30 Elo (post-Lc0-FT) | PILOT P3 |
| Label smoothing on policy | free | negative risk | REJECT |
| SAM | 2× step cost | ≈EMA for us | REJECT for now |
