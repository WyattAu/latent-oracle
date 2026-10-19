# Loop 3 — Training data quality: volume is binding, but the box cannot reach the knee

**Question in the loop plan:** SF-depth labels — d10 vs d16 both capped;
BC-v2 130M null. What label depth/diversity actually moves strength?

## 1. Our complete data-lever evidence base

| Lever tried | Volume | Result |
|---|---|---|
| d10 bundle, plain BC | 2M positions | **champion** (s1) |
| d16 labels + RCT/QAT | 1M positions | refuted (verdict B: −6.5 ± 12.3, 800 games) — confounded with recipe |
| d16 labels, plain BC | 1M positions | **this loop's match, running** (ops/label_depth.sh) |
| BC-v2 | 130M positions | null — volume did NOT save a thin recipe |
| ChessBench AV smoke | 4.6M samples / 20M records | puzzle 11.0%, bucket-acc 12.2% |
| ChessBench AV play | same | −225 ± 38 Elo vs champion (200 games) |
| Self-play winner moves | 10k / 30k games | null, null (s4 +2.32 ± 3.19, s5 −1.16 ± 3.93) |

## 2. The published scaling points (DeepNoc, GC-9M puzzle accuracy, approx.)

2K games ≈ 71% → 20K ≈ 76% → 100K ≈ 82% → 1M ≈ 86% → 10M games 88.9%
(onepass: 6.9M params, 1.15B examples, 24 A100-hours). Models ≥7M params
**overfit below 100K games** — the knee for our param class sits around
50–100M unique positions. We are at 0.4–4% of the knee.

## 3. The gap that volume does not explain

At DeepNoc's ~20K-game point (~10M positions, ~10k steps at batch 1024 —
compute comparable to our 12k-step run) their 9M net scores ~76% on the
same puzzles. Ours scores **11%**. Same data scale, same param class,
65-point gap. The difference is structural, not volume:

- **Encoding**: 64 bytes of piece-codes vs their rich per-square planes
  (+ castling/EP/rating features — our v3 additions were built for the BC
  trunk, unproven for AV distillation).
- **Head design**: our AV head pools the trunk state and scores a move
  embedding; DeepNoc concatenates state+action as a token the TRUNK reads
  (ChessFormer: same design, matches GC-270M at 30× less FLOPs).
- **Trunk init**: ours warm-starts from a best-move-BC trunk (may be a
  feature or a bug — untested).

Conclusion: at fixed data, **architecture moves the curve more than anything
left on the table for us**. That reorders the loop plan: loop 4 (head/encoding
design) is the productive next experiment, BEFORE any data scale-up.

## 4. Data-volume reality on this box (why the scale-up waits)

- One ChessBench shard = 1.5 GB ≈ 20M records. Disk free: 73 GB (92% full)
  → ~48 shards max ≈ 940M records.
- Decode: 10k records/s (measured). Train step (batch 384, 6 GB GPU): 1.4 s.
- One pass over 400M labels = 1.04M steps ≈ **17 days** on the RTX 2060 —
  infeasible. 130M records (4 shards) ≈ 340k steps ≈ 5.5 days — feasible
  only if the box stays idle, which it does not.

**Ruling:** no data scale-up on this box until (a) the architecture gap is
closed (loop 4) and (b) a GPU budget or smaller-param target exists. The
cheap lever order is: head/encoding redesign → 11%→(target ≥30%) on the
SAME 20M records → then data volume.

## 5. Verification

- The d16-vs-d10 plain-BC match (300 games, pair model) closes the label-depth
  question: expected null; a win would reopen verdict B's recipe question.
- Perpetual admission gate for data experiments: a new dataset/label recipe
  must beat the champion over 300 games (same protocol as ops/rl_gate.sh)
  before engine integration. BC-v2's 130M null already passed through this
  logic post-hoc; the gate makes it mandatory.

**MATCH RESULT (2026-10-18, final): NULL.** The 300-game "+36.0 ± 20.2"
was a false positive: extensions landed +12.2 ± 14.2 (600) then −4.7 ± 11.9
(900), pooling to **+7.7 ± 8.3 over 1800 games (0.93σ)**. Verdict B's null
stands on clean evidence: label depth (d10 → d16) does not move strength
through plain BC at 1-2M positions — and the RCT/QAT recipe question is
moot, since there is no gain for a recipe to mask.

**Protocol lesson (recorded in RESULTS):** a 1.78σ result is not a "live
lever". With multiple comparisons in flight, only ~2.5σ+ extensions justify
adoption. The extension rule worked as designed.

**Loop 3 verdict (final):** label depth is a null lever at ≤2M positions.
Volume is first-order but out of budget; the 11%-vs-~76% gap at matched
compute (loop 4's question, after the running compute-matched control)
is the binding constraint.
