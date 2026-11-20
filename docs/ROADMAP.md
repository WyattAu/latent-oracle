# Latent Oracle: Position, Path Forward, Roadmap
(2026-10-30; numbers from RESULTS.md, RESEARCH-HIGHLEVEL.md, the measured
SF-limit ladder, and the verified 2025-2026 literature)

## 0. THE DUAL-TRACK PROGRAM (2026-11-20 — supersedes the single-track state below)

The program maintains TWO first-class models from the same training stack,
because they serve different competitions and research goals:

### Track S — Searchless (policy-only)
- **Model**: policy argmax over legal moves, one forward pass, instant moves.
- **Champion**: repl_nodist_q (80M real-result pool x 3 epochs, no distill,
  INT8-served) — dethroned net_q_v2 by +170 in the sign-audited
  INT8-normalized ladder (RESULTS.md 2026-11-14/19).
- **Calibration**: ~1940 +- 50 absolute (SF-anchored, Nov 20: 42% vs
  SF@Elo-2000). The searchless ceiling is near for this architecture:
  sample scaling saturated at 240M samples; the remaining levers are
  POLICY-quality only (elite fine-tune, policy-head QAT).
- **Competitions**: searchless/chessbench-class benchmarks, the DeepNoc
  comparison line (2374 flagship to chase), human-alignment categories.

### Track AB — Net + alpha-beta search
- **Model**: the same net with SearchAB=true (alpha-beta over net-evaluated
  leaves, 400k nodes / 8s per move). One binary, one weights file, two
  serving modes via UCI options.
- **State**: the searcher's four logic bugs are fixed and validated
  (startpos-moves handler, ~uint8_t Color, is_draw 3-fold, node_cap parse;
  RESULTS.md 2026-11-19/20). Search over the CURRENT value head adds zero
  strength (ladder -489): the value head is calibrated to human-noise
  game results — it prices a captured rook at +120cp, indistinguishable
  from a quiet retreat, so the search cannot outplay the policy.
- **The unlock (running)**: fine-tune the value head on deep-Stockfish
  win_prob targets (the searchless_chess AV bags' win_prob field — the
  Lc0 thesis: clean value targets, then search adds real strength).
  ops/ft_sfvalue.sh: fine-tune -> INT8 -> ladders (AB-vs-searchless,
  vs champ_q, vs base). Complete ~Nov 20 late.
- **The second unlock**: NetQ::evaluate is ~122ms value-only (~8-20
  evals/s) — 20-50x slower than NNUE-class. The optimization (SIMD
  attention, quantization fusion, value-only path (done, -30%),
  threads-when-idle) is the follow-on once the clean-value ladder proves
  the search direction.
- **Competitions**: classical engine strength (the 2500 target), CCRL.

### The rules that keep the tracks honest
1. Ladders are INT8-vs-INT8 (or FP32-vs-FP32) — serving precision is part
   of the measurement (the sign audit, 2026-11-14).
2. Recipe selection happens AT serving precision: the distill inversion
   (+147 FP32 / -131 INT8) proved FP32 ladders select the wrong recipe
   for INT8-served engines.
3. Every keeper net keeps its .pt (re-exportable) and its INT8 (.bin).
4. Both tracks share the training stack; a fine-tune can feed both (the
   policy ladders decide whether Track S adopts it).

## 1. Where we are

(2026-10-30 state — HISTORICAL, numbers superseded by RESULTS.md
2026-11-14/20 and the sign audit; kept for the lever analysis below.)

Our champion (net_q_v2: 6.4M-param square-token transformer, single forward
pass, no search, INT8-served) measures **−129 / −193 / −252 vs the Stockfish
limit-strength ladder at 2000 / 2200 / 2400** — i.e. roughly the
**1700–1900 human-scale band**. Verified training state: the marquee (a
continuous pass over **526M human-move records**, 4 chained sub-slices) is
at slice 0 step ~61k/117k, match 0.525 and climbing.

### Comparison with the public record (verified sources in RESEARCH-HIGHLEVEL)

| Bot | Class | Strength | Our position |
|---|---|---|---|
| Stockfish 17 / Lc0-Big (T60/B60) | searched NNUE/RL, 3600+ CCRL | a different universe | by design: searchless |
| **Leela-CF-policy** (DeepNoc 2026) | searchless, 2.5–4M params — **our design class** | **2374 ± 37**, 93.5% ChessBench puzzles, 7.6B FLOPs | the gap to close; trained on 3.25M GM games + OnlineQAT |
| Tracking line (arXiv 2603) | sequence model, 120M | Lichess bullet **2570** | bigger net, human-move task shared |
| **Braun QAT net** (2610.09346) | searchless policy-only, 3.25M GM games | **1658–1717** policy-only; 3251 at 100k searches | **we measure at/above this line** (~1900 SF-limit) |
| **Maia-2 1900** | human-aligned | ~1900 FIDE-aligned | our band; our BC pool is the same philosophy at 4× scale |
| Lc0 small nets policy-only | searchless | well below their searched selves (+1,593 Elo with 100k searches) | our recipe: same trade, pushed further |

Honest summary: **we are inside the 2025–26 searchless-chess research wave**
(square-token transformers, from-to policy heads, human-alignment data) with
a champion already **at/above the Braun policy-only line and in Maia's
band**, and **behind DeepNoc's 2374 flagship** — whose recipe (3.25M GM
games + OnlineQAT + conv trunk) we have independently validated half of
(from-to head ✓, human-move data ✓, INT8/QAT line ✓ built) and whose
remaining deltas (GM-game data, QAT-on-policy, trunk style) are exactly the
queued roadmap items.

## 2. The five levers, ranked by (expected Elo)/(cost)

1. **Data pool** — the single largest measured lever this project has
   produced: target-type = **~375 Elo** (AV-as-BC −424 vs human-move −49 vs
   v2pool at identical exposure); pool scale = pending (marquee).
2. **INT8 serving** — the exporter + projection-QAT bonus: **+179–233**
   historically on bc40m; cheap, already built, applies to any new net.
3. **Multi-epoch exposure** — the champ's recipe: 150M × 3 epochs still wins
   every one-pass pool at 38M samples (+188 over bcpool today).
4. **Inference extras** — RecyclePasses=2 (+5-ish at 15+0.2), Syzygy ≤5men
   (perfect), both already implemented.
5. **RL fine-tune** — unbounded upside, gated (ops/rl_gate.sh), needs the
   EVR advantage signal (SF-Q hybrid) to avoid the collapse modes in the
   literature.

## 3. Roadmap

### Now → Nov 1 — Gate 1: the marquee verdict (RUNNING)
- 450M human-move pass (4 chained slices). Final ladder: bcpool450 vs
  champ (250 games) + vs SF-limit-2000 (250 games).
- **bcpool450 ≥ champ** → foundation = BC pool. Else → union-pool epochs
  (BC 450M + lichess 130M + AV 37M, mixed, 2 epochs).

### Nov 2–3 — Gate 2: INT8 the winner
- export_int8 + projection-QAT on the Gate-1 net → engine-ready blob.
- Ladder INT8-winner vs champ → expected **new champion** (+150–230 class).
- This is the release candidate.

### Nov 4–5 — Gate 3: engine-ready measurement
- RecyclePasses=2 + Syzygy ≤5men ON.
- Full SF-limit ladder re-measurement → the public number.
- DeepNoc-puzzle benchmark run (their public suite) → the first honest
  same-benchmark comparison vs Leela-CF's 93.5%.

### November — the intelligence phase
- **SF-Q hybrid EVR** (evaluator = SF at fixed nodes; policy = ours): the
  honest game-edge metric, and the advantage function for RL.
- **RL fine-tune**: grpo with EVR advantage, gated by ops/rl_gate.sh
  (300-game teacher-relative gate + puzzle sanity) — the loop the project
  was designed for.
- **Union-pool multi-epoch** if Gate 1 picks the merge branch.

### December — the class war
- At 450M+ human moves + INT8 + QAT, the param class has DeepNoc's 2374 in
  reach. Their remaining deltas: GM-game data (we have 526M human games —
  different, possibly complementary), OnlineQAT on the POLICY (our QAT was
  projection-only; policy-head QAT is the upgrade), conv/GAB trunk (the
  ablation queue, ~6h GPU each).

## 4. Standing risks

- **Restart storms** (mitigated: systemd units everywhere, resumable slices).
- **The marquee loses Gate 1** (mitigated: union-pool branch is ready).
- **INT8 bonus doesn't replicate on new nets** (then it was bc40m-specific;
  the release candidate is the FP32 winner at −0 bonus).
- **DeepNoc-puzzle benchmark exposes a data-type ceiling** (human moves may
  cap below GM-move nets on SF-best-move puzzles — if so, the two pools are
  complementary: BC for play, AV for puzzle/tactics; the union pool tests it).
