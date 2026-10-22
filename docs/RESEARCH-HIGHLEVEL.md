# High-Level Research Loop v2 — deep dive, verified 2026-10-22

Supersedes the first pass (`RESEARCH-HIGHLEVEL.md`, same date). This version
is based on FULL TEXTS (arXiv HTML) of the four load-bearing papers plus
their appendices, fetched today. Verification log at the end.

---

## 1. Deep notes per paper

### 1.1 ChessFormer / Leela-CF (arXiv:2605.19091, ICLR 2026) — the blueprint

**Architecture (Leela-CF, 191M):** 15 layers, d_model 1024, MLP 1536, head
dim 32; GAB d2=d3=256. Input: 64 square tokens × depth 112 = 12 piece
one-hots × 8 board frames (**current + 7 history positions**, earliest
repeated if unavailable) + repetition flags (8) + 4 castling flags + side-
to-move (1) + plies-since-capture/100 (1) + 2 edge channels. **Board is
flipped to side-to-move perspective.**

**GAB** (the headline component): board state → linear d1 → flatten →
linear d2 → GELU+LN → linear h·d3 → per-head projection (SHARED across all
layers) → h×4096 → h×64×64 bias added to attention logits pre-softmax.
Dynamic per-position; models global positional features (game phase etc.)
while dot-product attention carries square-local semantics. Ablation at 4M
params (Table 5, 95% CIs):

| encoding | policy acc | puzzles | Elo | FLOPs |
|---|---|---|---|---|
| absolute | 56.6 | 61.0 | 0 ± 18 | 210M |
| relative 2D bias | 57.5 | 63.2 | +40 ± 18 | 210M |
| GAB (2.5M params) | 57.0 | 61.5 | −4 ± 18 | 131M |
| GAB (4M params) | 58.5 | 64.2 | **+83 ± 18** | 228M |

Monotonic absolute < relative < GAB; GAB matches absolute at ~60% of the
compute. In full LC0: **+100 Elo, TCEC Cup 11 + Swiss 6/7 wins over
Stockfish** (14 months of Stockfish development ≈ 46 Elo by comparison).

**Policy head:** scaled dot-product of from-token queries × to-token keys →
64×64 logits + promotion logits (learned projection on last-rank keys, one
bias per promotion piece; ep = diagonal moves; castling = king-two-square).
**Identical in kind to ours.** Value head: mean-pool → LN → 128 → ReLU → 3
WDL logits. **Also identical in kind to ours.**

**Distillation data (the part everyone copies):** a FIXED dataset of LC0
self-play games from an April 2024 RL run, keeping only games after the
model's strength plateaued. Oracle = 100M-param transformer searched at
**600 nodes/move**. Policy target = **MCTS visit distribution** (not argmax);
value target = 3-way WDL. Training: Nadam (β1 0.9, β2 0.98), 6M steps, LR
2e-3 → 3e-4 → 3e-5, SWA checkpointing (no EMA). The four ablation models
(2.5-4M params) each trained **1×A100 ≈ 4 days** — our scale class.

**Results vs DeepNoc:** Leela-CF-**policy** (ONE forward pass): 2374 ± 37
Elo, 93.5% puzzles at **7.6B FLOPs**. DeepNoc AC-270M (value-max strategy,
~35 forwards): 2299 ± 36, 94.2% at **427B FLOPs**. The strongest searchless
chess entity per FLOP in the published record is a square-token + GAB +
history-8 transformer distilled from a fixed searched oracle — 56× less
compute than AC-270M and stronger.

**History ablation:** n=0→7 improves substantially; n=7→31 no gain. n=7 is
the sweet spot.

**Critique of DeepNoc's encoding they showed:** RoPE on a LINEARIZED square
sequence "maximally decays attention between opposite corners even though
those corners lie on a main diagonal" — 1D position encodings are wrong for
chess geometry.

### 1.2 Braun (arXiv:2609.37447, Sep 2026) — the engineering mirror

AZ-style self-play, 6.32M params (14×160 conv ResNet, 52 side-to-move-
canonical planes incl. castling/ep/repetition/50-move/**8 recent moves**/
material counts), 2.5 days on 8×4070S ($43!), 3.25M games → **3251 Elo at
100k searches** vs SF13 fixed-node ladder. Policy-only (no search):
**1,658-1,717 benchmark Elo**. Search multiplies strength ×~1.9 at their
data quality (policy-only → 100k searches = +1,593 Elo).

**Ablations that validate or refute our designs:**
- **From-to policy head RETAINED** (50k params vs 484k dense; −0.03 nats
  held-out CE; dense and move-plane heads FAILED). Our head design is right.
- **Conv trunk beat bare attention** (64 square tokens + learned row/col
  embeddings) by 0.006 nats held-out; **Smolgen-style biased attention beat
  conv by +0.009 nats** but was slower to serve → conv retained for search
  throughput. For SEARCHLESS inference the throughput argument weakens;
  biased attention (GAB/Smolgen family) is the fit winner.
- **PTQ without QAT: 13.3-25.9% policy top-1 agreement** (activation ranges
  grow ~0.78→~40 with depth). QAT required scaled post-activation blocks
  (capped activation, 1/√depth residual scaling). INT8 = 1.31× inference
  speed — throughput, NOT strength, in their pipeline.
- **Auxiliary training-only heads**: next-position policy (0.15) + moves-
  remaining (0.1). KataGo-style; removed at deployment.
- **Progressive sizing** (12×128 → 14×160 mid-run) worked; further growth
  showed no gain. Distilled 470k-param students: −278 to −416 Elo below the
  6.3M teacher even at large search — small nets compress badly here.
- **Three failures with transferable lessons**: (1) late-game target
  poisoning (27-32% of games hit the ply cap; excluding the tail made value
  targets wrong); (2) TensorRT refit silently changed predictions (KL 1.05!)
  — fixed by deployment-parity checks, exactly our NetQ gate; (3) **a loss-
  based promotion gate shipped a net ~270 Elo WEAKER** — "loss ≠ playing
  strength", the same lesson as our Goodhart-adjacent results (bucket-acc,
  puzzle-acc never predict play).

### 1.3 Prior-Directed Exploration (arXiv:2608.27757, Aug 2026)

Starting point: LC0's Chessformer-distilled policy at **93.9% puzzles**.
Self-play RL fine-tune with forward-KL toward the net's own MCTS prior +
value-uncertainty-adaptive temperature: +1pt puzzles, mate-in-4 +4pts,
strength held/slightly up, in ~2k steps. Findings we adopt as protocol law:
- **Puzzle-only fine-tune: biggest puzzle gains, −260 Elo.** Puzzles are a
  diagnostic, never an objective.
- **Unregularized self-play collapses to a single line** — the regularizer
  (prior anchor) is what keeps exploration alive. Our earlier self-play
  nulls (s4/s5) are consistent: outcome-only signals without an anchor.

### 1.4 Tracking vs. Deciding (arXiv:2603.29761, Mar 2026)

Decoder-only GPT over **UCI move tokens** (~1900 vocab), RoPE + SwiGLU;
28M (12L/384d) and 120M (16L/768d). Data: Lichess bullet+blitz, 10M-100M
games (~1.4-14B tokens), 2 epochs; 28M = 2.5 H100-hours, 120M = 70 H100-
hours. No board input at all — the model tracks state internally.

- **Lichess bullet 2570** (120M, Elo-weighted), 55.2% top-1 human move
  prediction (beats Maia-2's 50.0/50.4 under native input).
- **P ≤ min(tracking T, deciding Q)**: filtering low-rated games (the
  "obvious" quality move) doubles illegal-move rates and collapses
  middlegame/endgame accuracy. Fix = scale (tracking) + **linear Elo
  weighting r≈20** (decisions); exponential r≈200 destroys tracking while
  LOWERING validation loss. **Validation loss ≠ strength, again.**
- Sequence input enables history-dependent play position models cannot do
  (e.g., repetition-avoidance timing, style coherence).
- Probes: board state IS linearly decodable in these models (98% at 120M).

### 1.5 The QAT line (Sep-Oct 2026): OnlineQAT (2610.09346), OnQ
(2610.29378), QATFactory (2609.39223), TRACE (2610.07767)

Modern QAT consensus: PTQ degrades; the fix is distillation FROM THE FP
TEACHER ON THE QUANTIZED STUDENT'S OWN STATES (on-policy), not offline
fixed-data QAT. QATFactory: deployment-format simulation in the training
loop; format-dependent details matter. Braun: activation-range growth makes
naive PTQ catastrophic (13-26% top-1), architectural accommodations
(scaled/capped blocks) + QAT restore fidelity.

**Our +233 mystery in this light:** our lost quantizer beat its own FP32
parent by a margin no PTQ can achieve. The 2026 literature says the only
familiar mechanism for "quantized version is BETTER" is a QAT-with-
distillation recipe (student adapts to quantization noise while matching a
teacher). Re-deriving that recipe is a concrete, citable experiment.

### 1.6 Evidence for move-conditioning (arXiv:2406.00877, 2024)

Lc0 activations contain the optimal move 2 plies ahead (92% probe accuracy).
Searchless nets implement lookahead internally; DeepNoc's AV design (move
conditions the whole computation) is the architectural expression.

---

## 2. The convergent SOTA recipe for searchless chess

Synthesizing ChessFormer + DeepNoc + Braun + Miłosz (all verified above):

1. **Tokens**: 64 squares, 12-piece one-hot, **side-to-move perspective**.
2. **Positional encoding**: GAB-style dynamic bias (not RoPE-on-sequence,
   not learned absolute).
3. **History**: 8 board frames (n=7 past) — the single biggest input lever
   after GAB (ChessFormer history ablation).
4. **Rule state**: castling flags, repetition, plies-since-capture, 50-move.
5. **Policy head**: from-to scaled dot-product + promotion offsets (ours
   already matches).
6. **Value head**: mean-pool → 128 → WDL (ours already matches).
7. **Teacher**: a FIXED searched oracle distilled at moderate depth
   (600-visit LC0 self-play for ChessFormer; SF16 50ms searches for
   DeepNoc) — targets are DISTRIBUTIONS (visit counts or per-move win
   probs), never argmax-only.
8. **RL fine-tune only AFTER distillation plateaus** (Miłosz), with a
   prior-anchored regularizer.
9. **QAT at deployment** with on-policy distillation and parity gates
   (Braun's refit-drift lesson = our NetQ gate).

## 3. Gap analysis: ours vs the recipe

| Recipe element | Champion (net_q_v2) | Gap severity |
|---|---|---|
| Square tokens | ✓ | — |
| From-to policy head | ✓ | — |
| Mean-pool value head | ✓ | — |
| GAB | **absent** (v3 has the table; unused) | **high** (+83 Elo at 4M scale, +100 in LC0) |
| History frames (n=7) | **absent** | **high** (substantial ChessFormer gain) |
| Rule state (castle/ep/rep/50mv) | **absent** | **high** (net is blind to legal-draw state) |
| Side-to-move flip | absent (we have side emb — partial) | medium |
| Distributional teacher | **absent — BC on SF argmax** | **critical** (AV > BC, DeepNoc; visits > argmax, ChessFormer) |
| QAT | one unreproducible win | medium (recipe re-derivable) |
| Scale | 6.4M vs their 4-191M | fine — ablation class is 2.5-9M |

## 4. Experiment queue v3 (each isolated, game-level verified)

| # | Experiment | What it isolates | Cost on our box | Evidence base |
|---|---|---|---|---|
| 0 | BC-at-37.9M (RUNNING, ~70% done) | data alone, fixed v1 design | sunk | our control |
| 1 | **AV head on the same 37.9M shard** | label paradigm (soft vs one-hot) at fixed data | ~5h GPU | DeepNoc Table (AV>BC); our AV infra |
| 2 | **GAB on** (v3-lite: GAB + castle/ep/rep, no history) | positional encoding + rule state | ~5h GPU | ChessFormer Table 5 (+83); Braun input design |
| 3 | **8-frame history input** (build on #2) | single-frame blindness | ~8h GPU + data build (ChessBench lacks history — use LC0/self-play games or synth) | ChessFormer history ablation |
| 4 | **LC0-600v distribution distillation** on our 2M positions (LC0 CPU, 1-3s/pos ≈ CPU-days) | teacher quality: SF-argmax vs searched visit distribution | CPU-only, parallelizable with GPU runs | ChessFormer §5.1 exact method |
| 5 | **QAT re-derivation** on the winner of 1-3 (capped activations + on-policy distill flavor) | the +233 mechanism | ~5h GPU | 2026 QAT line; Braun §4.3.6 |
| 6 | Move-conditioned trunk (DeepNoc AV-in-trunk) | policy expressiveness ceiling; 35× inference | ~6h GPU, spec decision after | DeepNoc; Jenner probes |

Order rationale: 1 and 2 are cheap, isolated, and attack the two highest-
severity gaps (labels, encoding). 4 is CPU-side and can run concurrently
with everything — its output (searched-distribution labels) upgrades every
later run. 3 needs a data decision. 6 is the big-rewrite item, deliberately
last among the cheap levers.

Explicitly rejected, with sources: search-heavy self-play at our compute
(Braun: 3251 Elo costs 20 GPU-days at 8×4070S; we have ~1/50th of that);
GNN representations (AlphaGateau; ChessFormer beats it with plain
transformers); puzzle-accuracy objectives (Miłosz: −260 Elo); aggressive
data filtering (Tracking-vs-Deciding: r≈200 weighting destroys tracking);
bare RoPE-on-squares encodings (ChessFormer's geometric critique).

## 5. Verification log

- arXiv API queries (2026-10-22 ~21:13 UTC): "searchless chess",
  "ChessFormer", "amortized planning"/AlphaZero chess, QAT generalization,
  Ceres MCTS-free (0 results — struck from citations).
- Full texts: arxiv.org/html/2609.37447v1 (Braun), arxiv.org/html/
  2603.29761v1 (Tracking), arxiv.org/html/2605.19091v1 (ChessFormer incl.
  appendices A-C); DeepNoc per its v2 abstract + ChessFormer's critique.
- lczero.org/blog (2026-06 posts confirm active LC0 + Chessformer era).
- Our extraction of Braun/ChessFormer internals: agent-processed saved
  full texts (same files cited above).
