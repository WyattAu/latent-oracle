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

**High-level strategy (2026-10-22):** `RESEARCH-HIGHLEVEL.md` supersedes
the data-first path — verified 2026 sources (ChessFormer ICLR'26 + LC0,
Prior-Directed Exploration Aug'26, Tracking-vs-Deciding Mar'26, Braun's
6.3M->3251-Elo self-play paper Sep'26, the 2026 QAT line) reorder the
agenda to: label paradigm (AV soft targets) > GAB/rule-state inputs >
move-conditioned trunk > QAT re-derivation, with data scaling demoted to a
control arm. "Ceres" struck from citations (unverifiable).

**Trainer environment:** the persistent venv is
`/home/wyatt/.venvs/chess/bin/python` (torch 2.6.0+cu124, CUDA, python-chess).
`/tmp/opencode/venv` is NOT durable — `/tmp` gets wiped between sessions, so
all ops scripts point at the persistent path.

**Champion (2026-10-21):** `/home/wyatt/data/runs/bc/net_q_v2.bin` — the
INT8/QAT LOQW net (+136.0 ± 18.7 over av_v3/net_e1 at 3.1× size; RESULTS
2026-10-21). All future SPRT references should use it. Caveats: quantizer
provenance gap; both verdicts ran single-pass (RecyclePasses=1), so no
TC-depth confound.

## Loop Plan

| # | Part | Open problem | Papers | Other engines | Status |
|---|------|--------------|--------|---------------|--------|
| 1 | Move selection (searchless) | Single net eval picks moves; GRPO hacking showed the eval head is steerable. Can a pure policy play strong chess? | DeepMind "Grandmaster-Level Chess Without Search" (DeepNoc 2024); LC0 policy-only play | LC0, DeepNoc, Maia | CLOSED: data-starved at 0.004% labels (200-game AV play −225 Elo) |
| 2 | RL formulation | Static-eval rewards refuted (reward hacking); outcome-based RL needs the right formulation at affordable scale | AlphaZero/AlphaStack; Ceres (MCTS-free AZ); self-imitation learning | Ceres, LC0 self-play | PENDING |
| 3 | Training data quality | SF-depth labels: d10 vs d16 both capped; BC-v2 130M null. What label depth/diversity actually moves strength? | DeepNoc (9M GM games, no engine labels); Kasparov-era datasets | LC0 training drops, DeepNoc dataset | CLOSED: d16 label depth NULL after 1800 games (+7.7 +- 8.3, false positive caught); volume out of budget; loop 4 active |
| 4 | Net architecture | ChessNet v3 (GAB transformer, 6.5M params): board-image encoding vs move-history/state-action designs | DeepNoc bidirectional transformer; LC0 ResNet; five-column nets | LC0 arch survey | CLOSED: budget-bound (4 controls measured; 23M samples = 14.6% puzzles, −164 Elo) |
| 5 | Evaluation heads | Policy + value + WDL heads: which head layout drives play quality at fixed params? | DeepNoc Q-heads (state-action values); AZ value/WDL split | LC0 WDL, DeepNoc | CLOSED: exact tie, 11.0% puzzles both heads -- conditioning not the driver at 20M records; loop 4 inherits |
| 6 | Self-play generation | Loop proven (37 min/10k games); opening diversity and game-quality filters unexplored | CLOSED: diversity book-capped (~1980 distinct early positions / 30k games, 80-90% dups); fix specified, gated on a data-hungry target |
| 7 | Inference efficiency | Blob format v3 + INT8: quantization's strength cost measured? | CLOSED: INT8/QAT is a WIN, not a cost — new champion net_q_v2 (+136 over old champion, 3.1× smaller); TC-depth decomposition owed |
| 8 | Endgame | TB sidecar null at training; runtime TB paths in src/tb unexplored | CLOSED: runtime TB is a measured no-op in harness (150/150 splits); retained as correctness guarantee |
| 9 | Openings & variety | openings.epd + opening-weight 1.5: book diversity vs strength measurement noise | SPRT opening selection papers | CLOSED: paired-book + pentanomial infra audited sound; absolute Elo is book-conditional (rule recorded) |
| 10 | C++ engine integration | src/nn inference path: batch/blob plumbing vs the Python model — parity gates | CLOSED: NetQ parity gate built and passing (0/50); AV blob integration deferred per loop 4 |

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

### Loop 7 — closed: INT8/QAT dethrones the champion

Executing the stale Oct-4 int8_sprt chain produced the session's biggest
result: bc/net_q_v2 (INT8, 8.2 MB) scores +136.0 ± 18.7 (7.3σ) over the
reigning champion av_v3/net_e1, and +233 ± 30 over its FP32 parent-era net.
Fastchess's txt Elo line mislabeled match 1; the engine-name recount
protocol caught it (direct PGN count confirms). New champion recorded in
RESULTS. Caveats: quantizer-script provenance is a recorded gap
(sibling-session artifact), and a fixed-nodes match is owed to decompose
eval-quality vs TC search depth.

### Loop 6 — closed: self-play diversity is book-capped

Measured on the existing corpora: the generator seeds from the 2000-line
openings.epd and replays deterministically, so 30k games contain only
~1,970-1,982 distinct ply-8/12 positions (80-90% duplicate rate) — the
s4/s5 data was state-space-thin as well as outcome-thin. Fix specified
(unbounded seeding + root temperature 1.0 for 8-12 plies, gate: <10% dup
rate at 10k games), deferred until a training target can consume the
entropy (loop 4's budget verdict caps the AV target). Full write-up:
`RESEARCH-LOOP6-SELFPLAY.md`.

### Loop 4 — closed: budget-bound, architecture exonerated

The control chain: head conditioning (exact tie), 2× samples (+1.45),
warm-start (exact tie), convergence (60k steps: still rising, 14.6%
puzzles, −164 ± 28 play). The underfitting reframe corrected loop 3's
"flat scaling" artifact: the runs were unconverged, and the curve rises
log-linearly at +2.2 points/octave — reachable only with 30-100M+ samples
(3-10 GPU-days), i.e. out of this box's budget. The 65-point gap to
DeepNoc at matched params is compute, not architecture. Revisit trigger:
a GPU budget or a public checkpoint swap.

### Loop 5 — result: conditioning null (exact tie), loop 4 inherits

The full xattn run scored exactly the same as pooled: 220/2000 = 11.0%.
Smoke-scale conditioning gains do not survive to 20M records. The d16
extension also landed: combined +20 ± 11.5 over 900 games (1.75σ,
weak-positive, below the adoption bar). Loop 4's first obligation is a
compute-matched rerun (DeepNoc's 20K-game point ≈ 10M samples; ours was
4.6M) before encoding surgery can be credited.

### Loop 3 — final amendment: the d16 "borderline positive" was a false positive

The +36 ± 20 (300 games) collapsed through extensions (+12 ± 14 at 600,
−4.7 ± 12 at 900) to **+7.7 ± 8.3 over 1800 games** — a null. The extension
rule caught it exactly as designed; the lesson (1.78σ ≠ live lever; only
~2.5σ+ extensions justify adoption) is recorded in RESULTS. Label depth is
closed as a null lever; loop 4 (compute-matched control, then encoding) is
the active question.

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

### Loops 8, 9, 10 — closing sweep

- **Loop 8 (`RESEARCH-LOOP8-ENDGAME.md`)**: Fathom Syzygy root probing was
  already implemented (≤5 men, `SyzygyPath`); no verdict had ever enabled
  it. Measured TB-on vs TB-off with the new champion: perfect null
  (150/150 split pairs). Retained as a correctness guarantee.
- **Loop 9 (`RESEARCH-LOOP9-OPENINGS.md`)**: the 2000-line paired book +
  pentanomial + engine-name recount infrastructure is audited sound; twice
  this session it caught mislabeled/False-positive results. Absolute Elo is
  book-conditional — interpretation rule recorded.
- **Loop 10**: AV blob integration stays deferred (loop 4 budget verdict),
  but the deliverable that mattered now exists: `ops/netq_parity.py`
  (python quantized forward vs engine bestmove) — PASS 0/50 on both LOQW
  blobs after fixing the fuzz's own piece-code convention (shard format is
  black=pt+8). The champion's inference path is verified bit-faithful.

**Program state: all 10 loops closed.**
