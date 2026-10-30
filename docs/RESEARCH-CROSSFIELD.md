# Cross-Field Research Loops: transferring methods from quant, graphics,
# systems, and ML into searchless chess. (2026-10-30)

## 0. Measured bottlenecks that motivate the mining

- Target type: deep-SF-move BC is unlearnable per-sample (0.035 match) vs
  human moves (0.54) — **data representation ≫ architecture tweaks**.
- Lc0 probes show nets encode 2-ply lookahead internally, but searchless
  play cannot consistently EXPLOIT it — **lookahead without search**.
- The champion plays both colors from one white-POV convention — possible
  **hidden viewpoint asymmetry** (audited below: confirmed, large).
- The +233 INT8 effect = **quantization noise regularizes/improves
  inference** — our own accidental discovery, un-systematized.

## 1. Loop A — color-flip symmetry (graphics: consistency losses;
       algebra: equivariance) — **AUDITED: VIOLATION CONFIRMED**

Chess has an exact Z2 symmetry: mirroring ranks and swapping colors maps a
position to an equivalent one (castling rights map structurally). A perfect
evaluator must satisfy v(b) = 1 − v(mirror(b)) and return the mirrored best
move.

**Audit of net_e1 on 293 random plies:** mean |v(b) − (1−v(mirror))| =
**0.147 win-prob** (p90 = 0.28); policy top-1 agreement under mirror =
**10.2%** (chance-level for ~35 legal moves). The net treats mirrored
positions as unrelated — it learned side-specific style from the data's
temporal asymmetry (white moves first) instead of the geometry.

**Lever:** mirror augmentation at train time — replace each batch record
with its mirror with p=0.5:
  codes → rank-flip + color-swap (sq ^ 56, pt ↔ pt+8)
  side → flip; ep → ep ^ 56; castling → (r&3)<<2 | r>>2
  policy target → both squares ^ 56; promo unchanged
  wdl → white-POV swap (w,l) = (l,w); neutral unchanged
This doubles the effective data, enforces the equivariance inductive bias,
and cannot hurt (the symmetry is exact). Optional: explicit consistency
loss |v(b) − (1−v(m(b)))|.

**Expected effect:** the 0.147 violation is a *measured bias*; closing it
converts side-specific overfitting into generalizable strength. Mirrored
Lc0 training (canvas flips) is standard in the field — we never had it.

## 2. Loop B — one-ply Bellman consistency on game-streamed data
   (graphics: temporal coherence; RL: advantage identity)

The BC slices are game-ordered: consecutive records are consecutive plies.
For each adjacent pair (s_t, a_t, s_{t+1}):

  v(s_{t+1})_mover-conversion ≈ Q(s_t, a_t) = v(s_t ⊕ a_t)

An auxiliary loss |v(s_{t+1}) − v(s_t ⊕ a_t)| (both from the net, mover POV
converted) teaches **real one-ply lookahead without search** — the net is
rewarded for evaluating the afterstate consistently with its evaluation of
the position. Lc0's probe result (nets encode 2 plies internally) says the
capacity is there; this loss trains it out into the open.
Needs: game-boundary detection (halfmove reset / FEN diff), pair sampling.
Cost: one extra forward per pair, resumable data loader.

## 3. Loop C — entropy-gated adaptive compute (graphics: early ray exit;
       systems: speculative decoding)

Static RecyclePasses=2 for all positions wastes compute on quiet positions.
Gate a second forward on: (a) policy top-1 margin below threshold, or
(b) value-head entropy above threshold — both calibratable on held-out
games. Per-position compute allocation = strength-per-FLOP. Engine-side
only (uci.cpp), thresholds from the calibration corpus.

## 4. Loop D — distributional value head (quant: risk measures;
       Lc0: return buckets)

The (w,d,l) triple is a 3-point distribution. Lc0's V2/V3 use return
buckets (win-prob quantiles) and report stability gains. A 5–9 bucket head
(HL-Gauss targets from win_prob) gives sharper gradients on decisive
positions and better calibrated uncertainty — feeding Loop C's gate and
the EVR metric's uncertainty term.

## 5. Loop E — quantization noise as regularizer (OUR OWN finding,
       systematized)

The +233 INT8 effect = inference-time noise injection landed on a net
fragile in near-tie positions. Systematize: (a) train with quantization-
noise augmentation so the net is robust to its own serving format
(projection-QAT partially does this — extend to activations);
(b) measure: is the noise benefit reproducible as a *training* method
(add uniform noise to activations during training, serve clean)?
This inverts the accident into a method and explains the mechanism.

## 6. Loop F — opponent/phase conditioning (quant: regime models;
       Maia: opponent modeling)

FiLM-style conditioning of the trunk on (move number, material phase) or
(opponent proxy) — cheap adaptation without per-opponent nets. Related:
hypernet-generated biases; LoRA-style adapters per style band.

## 7. Priority order

A (implemented now — free, foundational) → C (engine-side, cheap) →
B (training-loop change, the novel one) → D → E → F. All are independent
of the marquee's scale verdict and compose with whatever Gate 1 picks.
