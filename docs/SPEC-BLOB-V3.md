# SPEC: Blob Format v3 — Single Bundle

**Status:** Design — implementation gated behind moonshot verdicts
**Decision:** one bundled format change (user, 2026-10-04), one
re-validation cycle. No further format churn after this until v4 is forced.

---

## 1. What goes in

| Feature | Source rationale | Params added |
|---|---|---|
| Castling-rights + ep inputs (3 tokens appended to state) | RESEARCH-DATA §3 (model currently infers from piece layout) | ~3×d |
| Material output buckets (8, by non-king piece count) | NNUE LayerStacks/buckets, RESEARCH-EXTERNAL §1.2 | value head ×8 |
| King-square bucket embedding (16 buckets) | NNUE king buckets | 16×d |
| Rating conditioning (16 buckets, 1800–3000) | Maia, RESEARCH-EXTERNAL §4 | 16×d |
| HiCo history tokens (last 3 plies: from,to,captured,check) | RESEARCH-NOVEL N1 | 4×3 move-embedding rows + zero-init gate g |
| GAB consumption in NetQ | blob v2 GAB currently FP32-only; v3 makes INT8 the single path | 0 |
| SCReLU option on value head | NNUE practice | 0 |

Zero-init/gating discipline (GAB trick, 4th use): every NEW input path
(history gate, rating/castling/ep contributions) is zero-initialized so a
v1/v2 checkpoint loads into v3 **bit-identically**. One warm-start story
for the whole bundle.

## 2. Format

```
header (unchanged 28 B): magic "LONW", version = 3, d, layers, heads, dff, dpol
tensor stream: v1 layout (unchanged prefix, strict prefix property kept)
  + [v2] gab_table (heads*8 f32)
  + [v3] castle_emb (6*d)      # K,Q,k,q rights as 4 one-hot + none/all  -> 6 rows
  + [v3] ep_emb (9*d)          # ep file or none
  + [v3] king_bucket_emb (16*d)
  + [v3] rating_emb (16*d)
  + [v3] hist_move_emb (13*d)  # 64from? no: 4 tokens/ply x 3 plies -> delta vocab ~13
  + [v3] hist_gate (1 f32, zero-init)
  + [v3] value_buckets (8 x value-head tail)
```
Engine Net.load reads version 3 after v1/v2 tensors; NetQ gains the same
(+VNNI-ready). Parity harness (loqw_sim) updated in the same commit;
CI tests: strict-prefix check, zero-init identity vs v1 weights, mirror
invariance including castling bits (K<->Q swap under file mirror!).

## 3. Subtleties called out now (so they don't bite later)
1. **Mirror vs castling bits**: file-mirror swaps K<->Q sides. The mirror
   augmentation must also swap castle bits (trainer-side), else labels lie.
2. **Rating conditioning leakage**: shard records lack ratings — derive a
   proxy per shard slice or drop the feature to v3.1 if data doesn't
   support it. Do not invent ratings.
3. **Material buckets change the value head's export shape** — keep_best
   and SPRT chains reference `net_eN.bin` paths, unaffected; but ALL
   existing v1/v2 blobs must keep loading (strict prefix guarantees it).
4. **NetQ v3**: GAB + new inputs must be integer-exact under the ADR-005
   contract (fp32 accumulation noise vs int8 boundary flips — parity
   tolerance stays argmax-stability, not bitwise scores).

## 4. Validation checklist (all green before any training run)
- [ ] strict-prefix: v3 blob ≥ v2 ⊇ v1 byte-prefix
- [ ] zero-init: v1 checkpoint in v3 model == v1 model outputs (max diff 0)
- [ ] CI: new tests for mirror+castling swap, bucket indexing, HiCo gate
- [ ] parity: engine FP32 v3 == python v3 (1e-6) on 6 probe positions
- [ ] NetQ v3 vs python v3: integer contract (argmax-stable)
- [ ] bench: eval latency regression < 10% vs v2 path
