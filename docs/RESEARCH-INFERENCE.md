# Research Notes — Inference-Time Techniques

**Scope:** getting more strength out of trained weights at play time, without
tree search. Ordered by implementation cost.

---

## 1. LoopCD — contrastive decoding over recurrent passes

**Paper:** "Decoding Looped Transformers Better for (Almost) Free" (Oct 2026)

### 1.1 Mechanism
If the trunk is run R times (recycling), pass r produces logits `L_r` from the
same input. Earlier passes are systematically weaker. LoopCD contrasts the
final pass with an earlier one:

- **LoopCD-Logits:** `L = L_R + α (L_R − L_r)` — one extra output projection.
- **LoopCD-Hidden:** contrast in hidden-state space, zero output overhead.

Reported: AIME pass@1 61.9% → 73.3% (Ouro-2.6B), HumanEval 22.6% → 31.7%
(Huginn) — training-free. Gains persist at half the loop count.

### 1.2 Chess adaptation
Our policy head outputs `64×64` scores. With `RecyclePasses=R`:

```
scores_R = policy(trunk_R(x))            # final
scores_1 = policy(trunk_1(x))            # saved from pass 1 (free)
scores   = scores_R + α (scores_R − scores_1)   # α ≈ 0.5–1.0
scores   = mask(scores, legal)
move     = argmax over from-to
```

- α is a UCI option (`LoopCDAlpha`, default 0.5), tunable via SPRT without
  retraining — a cheap post-release strength knob.
- Cost: ~0 extra compute (pass-1 policy is computed anyway).

### 1.3 Verdict: ADOPT with recycling (P1). ~40 lines in engine.

---

## 2. Recycling (adaptive computation)

**Papers:** LoopCD (above), "Scaling Laws for Looped Mixture of Experts"
(Sep 2026), "Looping Beyond Twice" (Oct 2026).

### 2.1 Design
Shared trunk, R forward passes. Between passes, feed the previous output
embedding back as input (optionally with a pass-count embedding).

### 2.2 Cost model at our scale
One trunk pass on a 64-token board ≈ 6.4M-param forward ≈ ~2 ms on RTX 2060
CPU inference / <0.5 ms GPU. R=4 stays far under our 60+6 budget.

### 2.3 Pass-count policy
Fixed R=2 is the safe default. Adaptive R (more passes when position is
"complex" — e.g., high pass-to-pass policy divergence) mirrors "conditional
computation" results but adds a tuning surface; defer.

### 2.4 Training recycling
Looped weights need looped training to be strongest, but LoopCD shows gains
even decoupled from training. Plan: train AV predictor normally (single
pass), then enable R=2 + LoopCD at inference and SPRT the combination. If
positive, add pass-noise during a subsequent fine-tune (`--recycle 2`) to
tighten the loop.

---

## 3. DiffuSearch — discrete-diffusion implicit search

**Paper:** arXiv:2502.19805 (see SPEC-DIFFUSION.md for full spec).

- Sequence: state, action, state, … (4 future plies). Bidirectional attention.
- Training: absorbing diffusion — mask random future tokens, predict them.
- Inference: T=64 denoising steps over the masked future; the first action
  token after denoising is the move.
- Reported +540 Elo over the same model's one-step policy.

Verdict: biggest single known gain; new paradigm. P2 pilot after the AV
predictor matures. Our SPEC-DIFFUSION.md already contains the tokenizer and
training changes; the open engineering item is batching T denoising steps
efficiently on a 6GB GPU (batch the steps, not the games).

---

## 4. Test-time training (TTT)

**Paper:** "JEPA-TTT: Persistent Test-Time Training of Latent World Models for
Planning under Dynamics Shifts" (Sep 2026); "Decision Titan" (Oct 2026);
"How Much Can Language Models Gain from Test-Time Computation?" (Oct 2026).

### 4.1 Idea
Update weights (or adapter weights) at inference time on the immediate
context, then predict. Powerful for distribution shift; the JEPA variants do
it in latent world-model space for planning.

### 4.2 Chess adaptation sketch
Fine-tune per-game on the opponent's revealed moves (a few gradient steps per
move — opponent modeling) before predicting ours. Cheap adapters (LoRA-rank-4
on value head) keep this ~free.

### 4.3 Verdict: WATCH (P4)
- Legality/safety: per-game adaptation is fine for engine rules.
- Risk: catastrophic drift over long games; needs eval gating per move.
- Only meaningful once we have a strong base (post-AV). Revisit at v0.5.

---

## 5. What we already run (no change)

- Syzygy DTZ/WDL probe for ≤5-man endgames (Fathom).
- Repetition/75-move gates in `bestmove_net_impl`.
- Low-clock PST fallback (<3 s).
- NetQ INT8 quantized inference (o=0 path verified).

## 6. Combined expected effect

| Inference stack | Est. Elo over BC-v1 baseline |
|---|---|
| BC-v1 one-pass | ~2050–2100 |
| + AV predictor weights | +100–200 |
| + Recycle R=2 + LoopCD | +50–150 |
| + DiffuSearch inference | +200–500 |
| **Ceiling estimate** | **~2400–2900** |

Honesty note: gains are not guaranteed additive; DiffuSearch may subsume
recycling (its 64-step denoising *is* iterative computation). Measure each
layer separately via SPRT before stacking.
