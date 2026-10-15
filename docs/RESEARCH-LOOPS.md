# Research Loops — Chess Spec Hardening Program

≥10 iteration loops over the chess repo's parts: for each part — pull papers,
analyze the cutting edge, compare against how other engines solve the same
problem (or establish it is unsolved), implement the best solution in the
spec + implementation, adjust the verification, repeat.

**Spec/proof mapping (chess vs formal projects):** the repo's "spec" is
`SPEC-BLOB-V3.md` (net format), the RESEARCH-*.md architecture documents, and
the implementation itself (C++ engine `src/`, Python trainer
`latent-oracle-data/trainer/`). There are no Lean proofs; the chess
equivalent of proof-adjustment is **verification by SPRT at game level**,
the engine-name recount protocol, and the perft/fuzz gates. Each loop
closes with those.

**Toolchain:** C++ engine (CMake/Ninja, 24-target build), Python trainer
(torch, Muon, mathlib-free), fastchess SPRT harness, the self-play loop
(37 min / 10k games).

## Loop Plan

| # | Part | Open problem | Papers | Other engines | Status |
|---|------|--------------|--------|---------------|--------|
| 1 | Move selection (searchless) | Single net eval picks moves; GRPO hacking showed the eval head is steerable. Can a pure policy play strong chess? | DeepMind "Grandmaster-Level Chess Without Search" (DeepNoc 2024); LC0 policy-only play | LC0, DeepNoc, Maia | IN PROGRESS |
| 2 | RL formulation | Static-eval rewards refuted (reward hacking); outcome-based RL needs the right formulation at affordable scale | AlphaZero/AlphaStack; Ceres (MCTS-free AZ); self-imitation learning | Ceres, LC0 self-play | PENDING |
| 3 | Training data quality | SF-depth labels: d10 vs d16 both capped; BC-v2 130M null. What label depth/diversity actually moves strength? | DeepNoc (9M GM games, no engine labels); Kasparov-era datasets | LC0 training drops, DeepNoc dataset | PENDING |
| 4 | Net architecture | ChessNet v3 (GAB transformer, 6.5M params): board-image encoding vs move-history/state-action designs | DeepNoc bidirectional transformer; LC0 ResNet; five-column nets | LC0 arch survey | PENDING |
| 5 | Evaluation heads | Policy + value + WDL heads: which head layout drives play quality at fixed params? | DeepNoc Q-heads (state-action values); AZ value/WDL split | LC0 WDL, DeepNoc | PENDING |
| 6 | Self-play generation | Loop proven (37 min/10k games); opening diversity and game-quality filters unexplored | AZ opening sampling; LC0 training-game filters | LC0 self-play | PENDING |
| 7 | Inference efficiency | Blob format v3 + INT8: quantization's strength cost measured? | INT8 quantization papers; LC0 quantized nets | LC0 INT8 | PENDING |
| 8 | Endgame | TB sidecar null at training; runtime TB paths in src/tb unexplored | Syzygy papers | Fathom, SF TB | PENDING |
| 9 | Openings & variety | openings.epd + opening-weight 1.5: book diversity vs strength measurement noise | SPRT opening selection papers | fishtest opening suites | PENDING |
| 10 | C++ engine integration | src/nn inference path: batch/blob plumbing vs the Python model — parity gates | — | — | PENDING |

## Per-Loop Protocol

1. **Research**: pull 4–6 primary sources for the part.
2. **Analyze**: extract what applies to THIS repo's exact design (searchless,
   6.5M-param GAB transformer, SF-distilled).
3. **Engine comparison**: Stockfish / LC0 / DeepNoc / Ceres / Dragon / Maia —
   solved better elsewhere, or unsolved everywhere?
4. **Implement**: spec (SPEC-*.md / RESEARCH-*.md) + code (engine/trainer).
5. **Verify**: build + perft/fuzz gates + SPRT at game level (the "proof").
6. **Record**: verdict line in the table + iteration notes below.

## Iteration Notes

### Loop 1 — Searchless move selection: AV distillation (model landed, scaling next)

**Research verdict (agent, sources read):** the DeepNoc claim is real but
scale-bound — 9M params reaches 2025±18 tournament Elo (2895 blitz vs
HUMANS only, 2299 vs bots), LC0 policy-only = 2292±16, and NOTHING at
6.5M params is published above ~1900-2100. ChessFormer's 6M policy matches
GC-270M at 30x less FLOPs (Shaw rel-pos + 64 square tokens + history).
Our 500k SF-labeled positions is 800-1000x less supervision than any
comparable net (DeepNoc: 15.3B action-value labels, public and free).

**The GRPO failure was predictable:** no successful searchless policy
trains against a static eval's argmax — DeepNoc distills a FIXED stronger
oracle (divergence structurally bounded), LC0/ChessFormer distill MCTS
visit distributions. RL on your own static eval is Goodhart with no
external anchor.

**Implemented (`chessbench_av.py`, data decoder `chessbench_data.py`):**
- move-conditional 128-bin win% AV head (HL-Gauss soft targets) over the
  GAB trunk's pooled residual stream (forward-hook capture)
- ChessBench .bag decoder: raw Beam TupleCoder records (NOT zstd — only
  .bagz is), varint-prefixed UTF-8 strings, 8-byte BIG-ENDIAN double
  win_prob (Beam FloatCoder), absolute-offset boundary index
- log-ratio clamp ±30 + torch.where finiteness masks in the GRPO trainer
  (the two NaN leaks, fixed and verified separately)

**First training run (2026-10-14):** 400k records (test split), 4k steps:
loss 4.90 → 3.79, bucket-acc 0.8% (random) → 9.1%, calibration 0.27 →
0.117, zero NaN incidents. 0.003% of DeepNoc's data — far left of their
scaling curve, which is still rising steeply at 1M+ games.

**Game-level verification (2026-10-14, 200/400 games):** the AV net
(puzzles 11.0% = 5.5x random) scores **21.5% vs the base net
(−225 ± 38 Elo)** via a Python UCI wrapper (argmax expected-win over legal
moves, ~35 forwards/move). At 400k records (0.004% of ChessBench) the AV
approach is decisively weaker than best-move BC -- DeepNoc's own scaling
curve (still steep at 1M games / 15.3B labels) and oneply's 1.15B-example
run say the fix is DATA SCALE (public, free) plus GPU-weeks, not
architecture. oneply's inversion result also matters: a HALF-TRAINED 7M
AV net GAINED +13.8 puzzle points from depth-2 search -- light search
compensates weak AV nets at our scale.

**Loop 1 verdict: the searchless single-eval design is data-starved, not
concept-broken. The scaling path is public (ChessBench), concrete (oneply:
6.9M net, 1.15B examples, 24 A100-hours), and deferred until a GPU budget
exists.** The engine keeps best-move-BC nets; the AV checkpoint and the
UCI wrapper (uci_chessbench_av.py) are preserved for the revisit.
