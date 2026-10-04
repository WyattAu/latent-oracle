# Research Notes — Novel Techniques (Invented for latent-oracle)

**Scope:** three original techniques designed from our stack's specific
gaps. Each entry: motivation, design, differentiation from prior art,
implementation map to our code, falsifiable test, expected value. Novelty
claims are post-prior-art-check and deliberately conservative.

---

## N1. History-Conditioned Searchless Policy (HiCo) — flagship

### Motivation
Every searchless chess net (Ruoss et al., Lc0-eval-mode, ours) conditions on
the **current position alone**. All game-flow information is discarded:
repetition pressure, 50-move-clock trajectories, transposition context,
opponent behavior over the game. Our engine papers over this with explicit
gates (repetition/75-move in `bestmove_net_impl`), but the POLICY itself
remains blind — it happily shuffles into a repetition the gate then has to
forbid, or misses that a "quiet" move is its third time-around maneuver.

### Design
- Input = current state tokens (67) ⊕ **last K=3 plies as move-delta tokens**
  (from, to, captured-flag, check-flag ≈ 4 tokens/ply → +12 tokens, +18%
  sequence length).
- The trunk must internally "unmove" the deltas (invertible with piece
  geometry — GAB's relation buckets are exactly the structure needed to
  route piece X from its origin square backward). This learned inverse-
  application is the mechanism worth testing.
- **Cold-start equivalence**: new move-token embeddings zero-initialized and
  a zero-initialized gate g on the history tokens' contribution → with g=0
  the model is bit-identical to the single-position net → any existing
  checkpoint warm-starts losslessly (same trick as GAB, ADR-quality).
- **History dropout** p=0.3 during training (zero the deltas) → the net
  stays robust when history is unavailable or OOD.

### Differentiation from prior art
- ESCHER (2022) uses game history for a *value/regret* function in
  imperfect-information games — not input-side history conditioning of a
  board-transformer policy.
- Lc0/Maia/AlphaZero family: single position. DiffuSearch conditions on
  FUTURE tokens; the PAST axis is unclaimed.
- Closest general idea: history stacking in Atari/DNN game agents (frame
  stacking) — pixel-space; the chess move-delta tokenization + unmove-
  learning is the novel part.

### Implementation map
- Trainer: shard pairs already bridge consecutive records (diffusion pilot's
  `find_connecting_move`) → history labels are free. Extend `encode_state`
  to emit `state ⊕ deltas`; blob v3 (castling/ep inputs land there anyway).
- Engine: UCI `position ... moves ...` already replays history
  (`apply_moves`); keep the last K moves in `Session` and tokenize them.
  ~30 lines in the tokenizer + inference path.
- SPRT: identical-position A/B, RecyclePasses=1, fast triage then official.

### Falsifiable expectation
+20–80 Elo (more at longer TCs where shuffle/steering matters), with the
mechanism check: replay-rate (move repetitions per game) should DROP
measurably in the HiCo engine's PGNs vs the base engine at equal Elo.
If replay-rate doesn't drop, the net isn't using history → kill.

---

## N2. Recycle-Consistent Training (RCT)

### Motivation
LoopCD/recycling gains at inference assume later trunk passes REFINE the
policy. Nothing in our training encourages that — the residual stream after
pass 1 is an arbitrary continuation point, so pass-2 policies can wander.
LoopCD is training-free by design; the training-side complement is
unclaimed territory.

### Design
For R recycling passes during training:
```
L = CE(final pass outputs)                    # standard loss
  + λ Σ_{r<R} KL( sg(policy_R) ‖ policy_r )   # pull EARLIER passes toward
                                              # the final (stop-grad on final)
```
- Direction matters: earlier passes move toward the final, never the
  reverse (that would blur the authoritative output).
- λ ≈ 0.5, R=2 (one extra trunk pass ≈ 2× step cost — fine-tune stage only,
  not pretrain).
- Connection: mean-teacher/self-distillation consistency, applied to
  *inference-time recurrent passes* — the novel instantiation. The looped-
  transformer literature (LoopCD, Ouro, Huginn) trains loops with noise or
  depth objectives; none trains pass-consistency as an inference-alignment
  objective.

### Implementation map
- Trainer: needs per-pass policy extraction — a `forward_recycle(ids, R)`
  on DiffuNet/ChessNet-style trunks returning all passes' scores. ~25 lines.
- Engine: already supports RecyclePasses/LoopCDAlpha (implemented this
  session) — zero engine changes.
- 2×2 experiment: {RCT, no-RCT} × {R=1, R=2+LoopCD} → 4 SPRT legs.
  RCT must not hurt R=1 (sanity) and must amplify R=2 (hypothesis).

### Falsifiable expectation
+20–60 Elo for RCT@R=2 vs no-RCT@R=2, and R=4 becomes viable. Kill
criterion: RCT@R=1 within noise of no-RCT@R=1 but RCT@R=2 also within
noise → the consistency loss isn't shaping pass dynamics → drop.

---

## N3. Uncertainty-Gated Adaptive Recycling (UGAR) — sketch

Use the pass-disagreement signal itself as a per-position compute dial:
`gate = KL(policy_1 ‖ policy_2)`; positions above a calibrated threshold
get extra passes. The threshold calibrates on the puzzle suite (cheap,
paired) and ships as a UCI option. Strictly a post-RCT refinement: only
meaningful if RCT makes later passes meaningful. One line of novelty: the
gate is *self-generated by the recycling dynamics* — no auxiliary
uncertainty head, no SF at inference. Estimated +10–30 Elo at ~30% average
compute reduction vs fixed R=4. Test after N2 verdict.

---

## Prior-art check outcome (honesty log)

| Candidate idea | Verdict |
|---|---|
| History-conditioned searchless policy (N1) | No direct prior art found (ESCHER is value-side, different) → **claim as novel design** |
| Recycle-consistent training (N2) | No prior art found on pass-consistency objectives → **claim as novel design** |
| Adaptive recycling gate (N3) | Adaptive computation is a broad field; the self-generated gate on recycling passes is a small delta → sketch, not claimed |
| Teacher-anchored GRPO (my earlier idea) | **Already exists**: "When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibration" (Sep 2026) → cite in RESEARCH-RL.md, do not claim |

---

## Build order

1. **N2 first** — zero engine changes, reuses the AV fine-tune stage, 4-leg
   SPRT slots into the existing chain tooling.
2. **N1 second** — rides blob v3 (castling/ep/rating-conditioning land
   together), engine change ~30 lines, needs the history-dropout training
   variant.
3. **N3** — only after an N2-positive verdict.
