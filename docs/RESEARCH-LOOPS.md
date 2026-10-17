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

**Trainer environment:** the persistent venv is
`/home/wyatt/.venvs/chess/bin/python` (torch 2.6.0+cu124, CUDA, python-chess).
`/tmp/opencode/venv` is NOT durable — `/tmp` gets wiped between sessions, so
all ops scripts point at the persistent path.

## Loop Plan

| # | Part | Open problem | Papers | Other engines | Status |
|---|------|--------------|--------|---------------|--------|
| 1 | Move selection (searchless) | Single net eval picks moves; GRPO hacking showed the eval head is steerable. Can a pure policy play strong chess? | DeepMind "Grandmaster-Level Chess Without Search" (DeepNoc 2024); LC0 policy-only play | LC0, DeepNoc, Maia | CLOSED: data-starved at 0.004% labels (200-game AV play −225 Elo) |
| 2 | RL formulation | Static-eval rewards refuted (reward hacking); outcome-based RL needs the right formulation at affordable scale | AlphaZero/AlphaStack; Ceres (MCTS-free AZ); self-imitation learning | Ceres, LC0 self-play | PENDING |
| 3 | Training data quality | SF-depth labels: d10 vs d16 both capped; BC-v2 130M null. What label depth/diversity actually moves strength? | DeepNoc (9M GM games, no engine labels); Kasparov-era datasets | LC0 training drops, DeepNoc dataset | CLOSED: volume binding but out of budget; 11%-vs-~76% gap at matched compute makes loop 4 first |
| 4 | Net architecture | ChessNet v3 (GAB transformer, 6.5M params): board-image encoding vs move-history/state-action designs | DeepNoc bidirectional transformer; LC0 ResNet; five-column nets | LC0 arch survey | PENDING |
| 5 | Evaluation heads | Policy + value + WDL heads: which head layout drives play quality at fixed params? | DeepNoc Q-heads (state-action values); AZ value/WDL split | LC0 WDL, DeepNoc | IN PROGRESS: xattn-vs-pooled head run chained (gate: 30% puzzles) |
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

### Loop 5 — Evaluation heads: xattn conditioning experiment (running)

Full write-up: `RESEARCH-LOOP5-HEADS.md`. Loop 3's 65-point gap analysis
points at head conditioning first: the pooled head mixes move and state only
in the last MLP; the xattn head lets the move embedding attend over all 64
board tokens for +0.4M params and no change to the one-trunk-pass inference
contract. Smoke evidence at 500 steps: 6.5% vs ~3.5% bucket accuracy.
Full 20M-record run chained behind the label-depth match
(`ops/xattn_after_match.sh`); gate for engine integration: ≥30% puzzles.

### Loop 3 — Data quality: ordering decision (volume waits, architecture first)

Full write-up: `RESEARCH-LOOP3-DATA.md`. The d16-vs-d10 plain-BC match
closes label depth as second-order at ≤2M positions. The decisive number:
at DeepNoc's ~20K-game compute point their 9M net scores ~76% on the public
puzzles while our AV net scores 11% — same data scale, same param class.
Encoding (64-byte piece codes vs per-square planes) and head design (pooled
state + move embedding vs state-action token read by the trunk) are the
structural suspects. One pass over 400M ChessBench labels is a 17-day RTX
2060 run: out of budget; 130M is 5.5 idle-box days, contingent on (a).

**Loop 3 verdict:** volume is the first-order constraint but cannot be paid
on this box; the head/encoding redesign (loop 4) at FIXED data is the
binding, affordable experiment. Data scale-up is deferred behind loop 4's
puzzle-accuracy gate (≥30% on the same 20M records).

### Loop 2 — RL formulation: closed by measurement (spec ruling + gate)

Full write-up: `RESEARCH-LOOP2-RL.md`. The literature and our own runs agree
on one structural fact: every successful searchless policy learns from a
target *better than itself* (MCTS visit distributions in AlphaZero, SF search
statistics in DeepNoc/ChessFormer/Dragon). Remove the search and remove the
target; what remains is a policy graded by its own or a co-trained head,
which is exactly our GRPO failure mode (KL 371 nats, reward ceiling by step
~300, 15/15 checkpoints losing to their teacher) and our self-imitation
nulls (s4 `+2.32 ± 3.19`, s5 `−1.16 ± 3.93`).

**Deliverables (spec + enforcement, not a new training run):**
1. Spec ruling: the blob carries no RL artifacts; move selection stays a
   deterministic single-forward argmax; training is supervised by default.
2. `ops/rl_gate.sh` — an RL run may touch the engine only if it beats *its own
   BC teacher* over 300 games at tc=15+0.2, pair-model, engine-name recount,
   with the 95% lower bound above −25 Elo (a null is not a pass). GRPO fails
   this by construction, which is why it is refuted rather than tuned.

**Loop 2 verdict:** RL formulation is settled for this repo — distillation
from a fixed oracle is the only formulation that yields gains here. RL's
legitimate role is game generation at 10²–10³× our current volume, not
policy optimization on this box.

**Loop 1 verdict: the searchless single-eval design is data-starved, not
concept-broken. The scaling path is public (ChessBench), concrete (oneply:
6.9M net, 1.15B examples, 24 A100-hours), and deferred until a GPU budget
exists.** The engine keeps best-move-BC nets; the AV checkpoint and the
UCI wrapper (uci_chessbench_av.py) are preserved for the revisit.
