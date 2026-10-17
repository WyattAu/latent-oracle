# Loop 5 — Evaluation heads: does move conditioning drive play quality at fixed params?

**Question in the loop plan:** policy + value + WDL heads — which head
layout drives play quality at fixed params?

## 1. What the engines/papers chose

| System | Head layout | Move selection cost |
|---|---|---|
| LC0 | policy planes (from/to) + WDL scalar, ONE trunk pass | 1 pass |
| AlphaZero | policy planes + value scalar | 1 pass (then MCTS) |
| DeepNoc AV | state-action value: move conditions the representation; 128 win bins | ~35 full passes |
| ChessFormer | move-token read by the trunk (policy) | 1 pass |
| **Ours v1 (pooled)** | move embedding concatenated with mean-pooled state, MLP → 128 bins | 1 trunk pass + 35 cheap head passes |
| **Ours v2 (xattn)** | move embedding attends over the 64 board tokens, then MLP → 128 bins | same cost class |

DeepNoc's own tables put AV-style nets ~200 Elo above policy-style nets at
9M params (88.9% vs 65.7% puzzles, policy-only LC0 at 2292 vs AV 2400+ at
comparable scale) — but their AV nets pay 35 FULL trunk passes per move.
Our head design keeps the spec's ~one-trunk-pass contract and tries to buy
the conditioning effect with one cross-attention layer.

## 2. The experiment (running)

`ops/xattn_after_match.sh` (chained behind the label-depth match) trains the
xattn head on the SAME 20M records, same 12k steps, same batch, same init
(av_s4) as the pooled av_s6 run — a controlled single-variable comparison.

- Smoke evidence (400k records, 500 steps): xattn bucket-acc 6.5% vs pooled
  ~3.5% at the same step — the conditioning effect is real at the margin.
- **Gate:** ≥30% puzzle accuracy (vs pooled's 11.0%) before any engine
  integration work (loop 10). 30% is chosen as the point where the scaling
  curve suggests play strength crosses the BC champion.

## 3. Decision tree

- **xattn ≥ 30% puzzles:** head conditioning is the driver → loop 10 (C++
  AV inference: 1 trunk pass + 35 cross-attention readouts) → SPRT vs
  champion → possibly reopen DeepNoc-scale data (loop 3's deferred plan).
- **xattn ≈ 11%:** conditioning is NOT the bottleneck → the 65-point gap is
  encoding (64-byte piece codes vs per-square planes) → loop 4 trunk/encoding
  surgery is the next experiment.
- **xattn between:** both contribute; measure, then loop 4.

## 4. Spec impact

- `SPEC-BLOB-V3.md` gains an AV extension only if the gate passes: the blob
  would carry the trunk plus the xattn head (0.4M params, ~2% size increase);
  the inference contract stays "one trunk pass per position".
- If the gate fails, the blob stays BC-policy-only and loop 4 inherits the
  open question.

**Loop 5 verdict:** pending the xattn run (~5h after the label-depth match).
