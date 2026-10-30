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

## 2.1 Loop B baseline measurement (2026-10-30, bcpool net)

corr(v(s), minimax-1ply(s)) = **0.42** on 249 held-out slice0 positions,
where minimax-1ply(s) = max_a (1 − v_opp(s⊕a)) over all legal afterstates
computed with the same net. The net's static value captures less than half
of the linear signal available from its own one-ply search — **the
self-consistency gap (RESEARCH-EDGE §2.2) is large and measurable**.

The CPU fine-tune proof was invalid (afterstate targets computed under
no_grad — no gradient path). The proper ablation: `--distill1ply` in
train.py — afterstate batch through the graph, stop-grad max as the value
target, MSE loss at weight λ≈0.1. Cost ≈ 35× forwards on the distilled
fraction; queue behind the marquee (GPU-bound).

## 3.1 Loop C measured (2026-10-30, bcpool net, 513 held-out slice1 positions)

- Margin predicts RP1→RP2 move-instability near-perfectly: change-rate
  39.8/24.5/7.8/2.0/0.0% across margin quintiles (entropy: 1→24%).
- **But on changed moves, RP2's move matches the human target only 18.4%
  vs RP1's 47.4%** — the second pass moves AWAY from the trained policy.
- Gating simulation: every threshold loses to always-RP1 (54.6% target
  match vs ≤51.9% for the best gate).

Verdict: for single-pass-BC nets the second pass is not weak search — it's
an untrained policy drift. RecyclePasses=1 stays the default for BC-pool
nets. The margin gate + this harness remain available for game-level Elo
tests per served net (target-match ≠ Elo, but the burden of proof shifted).


## 8. Data-scaling extrapolation for the BC pool (2026-10-30)

DeepNoc BC-9M: 65.7% puzzles (670M-position bag). Our bcpool: 34.7% at 38M
samples; v2pool: 43.0% at 38M (lichess distribution). The marquee passes
~450M BC records (12× bcpool's exposure, 2/3 of DeepNoc's bag).

Prediction (log-linear interpolation between our 38M point and DeepNoc's
670M point, same bag): puzzle accuracy ≈ 34.7 + (65.7−34.7) × ln(450/38) /
ln(670/38) ≈ 34.7 + 30.9 × 2.47/2.87 ≈ **61%** ± 4.
The Elo-scale verdict (vs champ) is Gate 1's ladder; the puzzle number is
the same-benchmark sanity check. If the marquee lands ≪61%, the BC bag's
later slices are redundant (game-ordered data → slice0-1 already cover the
distribution) and the union-pool branch takes over.

## 9. Competitive landscape sweep (2026-10-30): the direct competitor

**mateuszgrzyb-pl/searchless-chess** (GitHub, 8★) — the closest project to
ours found in the field sweep: solo researcher, value-based searchless move
selection (evaluates positions with deep-Stockfish cp, argmax over moves —
our exact design), ViT/ResNet models, 316M deduplicated lichess+SF
positions, 200+ hours on an A100.

| model | params | claimed ELO | ELO/1M params |
|---|---|---|---|
| ResNet-XL | 24.7M | 1719 | 70 |
| **ViT-Small** | 2.64M | **1817** | 688 |
| **ViT-Medium** | 9.5M | **1960** | 206 |

Scale caveat: their ELO is **fitted from Lichess puzzle-tier accuracy**
(1200 puzzles, 12 tiers), NOT engine matches — not directly comparable to
our SF-limit ladder (−129/−193/−252 at 2000/2200/2400). Still: same band,
same approach, solo scale — **our closest peer**.

Assets: models public (Keras .zip releases), dataset public on HuggingFace
(**mateuszgrzyb/lichess-stockfish-normalized**, 316M positions as parquet).
Notable: their dataset is the OLD-POOL STYLE (lichess + SF cp values) —
independent confirmation that human/cp data is the trainable target type.

Actions taken: none yet. Options: (a) port their ViT-Medium weights into a
ladder-able wrapper for a true engine-match comparison (Keras→torch port,
medium effort); (b) union their 316M cp-annotated positions with our
lichess pool for a Gate-1-merge-branch scale-up; (c) adopt their
puzzle-tier evaluation as a secondary public-facing metric.

## 10. Loop G — value-max play: FALSIFIED (2026-10-30, decisive)

Implemented the DeepNoc AC-line strategy in the engine (ValueMax UCI
option: evaluate every legal afterstate with the value head, play the move
minimizing the opponent's expected score; ~N forwards per move). Ladder:
**champ-policy +441 ± 54 over champ-value-max** (75 pairs, 92.7%, zero
losses; value-max is also 12× slower per move).

Why it fails here and worked for DeepNoc: our value head is trained on
*game outcomes of human games* — it predicts who wins a position, not
which move is best. Ranking afterstates by it re-asks the policy question
with a worse instrument (the afterstate distribution is off-policy). Deep
Noc's value-max worked because their AV head was trained specifically for
move evaluation (action-conditioned values from deep search).

Consequence: the road to 2500 runs through **data + policy quality**
(marquee → Gate 2 INT8 → GM-move data, policy-head QAT, AV head trained
for move evaluation), not inference-time tricks. The ValueMax option stays
in the engine as a research switch.
