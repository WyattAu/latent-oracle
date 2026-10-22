# High-Level Research Loop — verified against live sources, 2026-10-22

Every claim below was pulled from arXiv/lczero.org on 2026-10-22 (not from
model memory). Sources checked via arXiv API queries and the LC0 blog.

## 1. The verified landscape (what actually changed since our loop-1 notes)

### The architecture family we half-belong to
- **ChessFormer** (Monroe & Chalmers, arXiv:2409.12272, Sep 2024): square
  tokens + domain position encoding in attention; outperforms AlphaZero at
  8x less compute, prior GM transformers at 30x less.
- **ChessFormer: A Unified Architecture** (Monroe, Eilender, Chalmers, Tang,
  Anderson, ICLR 2026, arXiv:2605.19091): adds **Geometric Attention Bias
  (GAB)** and an attention-based source-destination policy head. Integrated
  into LC0: **+100 Elo, tournament wins over Stockfish**. Maia-3 human move
  prediction 57.1% top-1. **Our v3 `gab_table` descends from this line —
  and the champion (a v1 net) has it DISABLED.**
- **AlphaGateau** (Rigaux & Kashima, arXiv:2410.23753, 2024): graph/GAT
  representation, order-of-magnitude faster small-param training. (Superseded
  for our purposes by ChessFormer's results with plain transformers.)

### The searchless-distillation line
- **DeepNoc** (Ruoss et al., arXiv:2402.04494 v2, updated Oct 2024):
  ChessBench, 15B SF16 labels; target ordering **action-values > behavioral
  cloning > state-values**; 2895 blitz vs humans / 2299 vs bots at 270M.
  Confirms our loop-1 numbers.
- **Prior-Directed Exploration for Searchless Chess** (Miłosz, Duch,
  Grabowski, arXiv:2608.27757, Aug 2026): fine-tunes LC0's Chessformer-
  distilled policy (starting at **93.9% puzzle accuracy**) with self-play RL
  using forward-KL toward the net's own MCTS prior + value-uncertainty-
  adaptive temperature: 93.9->94.9% puzzles, mate-in-4 77->81%, strength
  held. Two warnings we should frame on the wall:
  1. **A puzzle-only fine-tune gained the most puzzle accuracy and LOST
     ~260 Elo.** Puzzle accuracy and playing strength dissociate.
  2. **Without a regularizer, self-play collapses onto a single line.**
  Their lever applies AFTER distillation reaches high accuracy — at our
  11-15% it is premature. It also retroactively validates our RL gate.
- **Tracking vs. Deciding** (Li & Jiang, arXiv:2603.29761, Mar 2026):
  sequence (move-HISTORY) input, trained from HUMAN games only, no engine
  labels; 120M params searchless -> **Lichess bullet 2570**; formalizes
  P <= min(tracking, deciding); low-rated games needed for tracking,
  Elo-weighted loss for decisions.

### The existence proof at our scale (the paper our "oneply" notes pointed at)
- **Engineering Efficient Self-Play Chess** (Braun, arXiv:2609.37447, Sep
  2026): AlphaZero-style from scratch on a SINGLE 8-GPU node in 2.5 days;
  **6.32M params -> 3251 benchmark Elo @ 100k searches/move vs an SF-13
  ladder**; 3.25M games, 100B simulations; documents policy-representation
  ablations and an explicit list of alternatives that failed. Code + model
  artifacts public. Same param class as ours: with search, +1350 Elo over
  our searchless champion. The gap is search + self-play volume, not
  architecture size.

### Interpretability (why move-conditioning is principled)
- **Learned Look-Ahead in Lc0** (Jenner et al., arXiv:2406.00877, 2024):
  probes read the optimal move 2 plies ahead from Lc0's activations at 92%
  accuracy — policy nets implement lookahead internally; giving the trunk
  access to the candidate move is the architectural expression of this.

### QAT line (the likely explanation of our +233 mystery)
- Modern QAT is **on-policy distillation from the FP teacher on student-
  visited states** (OnlineQAT, arXiv:2610.09346; OnQ, arXiv:2610.29378;
  QATFactory, arXiv:2609.39223 — all Sep/Oct 2026). QAT is consistently
  reported to IMPROVE over FP on quality, not merely match it. Our lost
  quantizer (+233 over its own FP32 parent, +136 over the old champion)
  is plausibly exactly this recipe applied at 8-bit — worth re-deriving.

### Corrections to our own ledger
- **"Ceres (Inria, MCTS-free AZ)" — NOT FOUND on arXiv.** Earlier references
  in RESEARCH-LOOP2-RL.md and the loop table were from model memory, not
  verified sources. Struck from the citation set; the RL-formulation
  conclusions of loop 2 stand on AlphaZero ablations + our own measurements,
  which is sufficient.

## 2. What this means for OUR design (the clear picture)

The champion is a v1 net: 64 square tokens (piece+square+side only),
bilinear from-to policy head, one forward pass, BC on SF argmax of <=2M
positions, then an unreproducible 8-bit QAT step. High-level gaps vs the
verified state of the art:

| # | Axis | Ours | SOTA practice | Expected effect |
|---|------|------|---------------|-----------------|
| 1 | Label paradigm | one-hot argmax (BC) | soft per-move win-probs (AV, DeepNoc) | same data, ~35x more information/record; DeepNoc: AV > BC |
| 2 | Move conditioning | trunk blind to move; move only in final dot product | move-conditioned scoring (DeepNoc AV; Jenner's lookahead evidence) | the policy head's expressiveness ceiling |
| 3 | Positional encoding | learned square emb only | GAB (ChessFormer ICLR'26; +100 Elo inside LC0) | our v3 has the table; the champion doesn't use it |
| 4 | Rule-state inputs | castling rights, ep, 50-move INVISIBLE | standard inputs everywhere | free correctness; tiny param cost |
| 5 | History | single frame | sequence input -> Lichess bullet 2570 from human games alone | different strength profile; free data route |
| 6 | Search | none | 100k searches -> 3251 Elo at our param count (Braun) | the ceiling is search+data, not size |
| 7 | QAT | one unreproducible win | on-policy distillation QAT (2026 literature) | possibly the +100-class lever we already stumbled into |

## 3. Revised experiment queue (high-level first, data second)

0. **Control (running):** BC-at-37.9M, v1 recipe — answers "data alone at
   fixed design" from below.
1. **AV-paradigm on the same 37.9M shard** (soft win-prob targets): isolates
   label paradigm at fixed data. DeepNoc's ordering predicts AV > BC.
2. **GAB + castle/ep on the BC recipe** (v3-lite from scratch): aligns us
   with ChessFormer ICLR'26 inputs; fixes the rule-state blindness.
3. **Move-conditioned trunk** (move token through all blocks): the DeepNoc
   AV design done faithfully; 35x inference cost, priced after measuring.
4. **QAT re-derivation** on whichever net wins 1-3 (on-policy distillation
   flavor per the 2026 literature), targeting the +100-class effect.
5. **Sequence/history input from Lichess human games** (free data, the
   Tracking-vs-Deciding route) — bigger build, defer behind 1-4.
6. **LC0-policy distillation** (soft, MCTS-shaped teacher instead of SF
   argmax) — the label-quality endgame; needs LC0 inference per position,
   priced later.

Explicitly NOT pursued: search-heavy self-play at our compute (Braun needs
20 GPU-days for 3251; we have ~1/50th of that), GNN representations
(superseded), puzzle-accuracy fine-tuning as a goal (the -260 Elo warning).

## 4. Protocol note
Puzzle metrics are reported but never optimized as objectives (Miłosz's
dissociation result). All verdicts remain game-level SPRT with the
engine-name recount.
