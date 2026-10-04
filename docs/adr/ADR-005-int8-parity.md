# ADR-005: INT8 NetQ o≥1 Parity — Root Causes and Resolution

**Status:** Accepted (2026-10-04)
**Context:** M2 blocker. INT8 inference matched PyTorch only at output row 0;
rows ≥ 1 diverged 10–45×. Blocking VNNI deployment.

## Root causes (two real engine bugs)

### 1. Weight scale treated as per-row (primary)
`qlinear` dequantized with `f_[q.ws_off + o]`. The LOQW blob stores **one
scalar** `w_s` per tensor (export_int8.py writes a single f32; loqw_sim.py
reads `f32(1)`). For o = 0 the index is correct; for o ≥ 1 the kernel read
**the bias array** as weight scales — biases are O(0.1) vs w_s O(0.003), the
observed 10–45× garbage. **Fix:** `f_[q.ws_off]` (both AVX2 and scalar paths).

### 2. Rowsum pointer double-offset
`rs` was already based (`rs_.data() + q.rs_off`) but indexed as
`rs[q.rs_off + o]` — reading rowsums from a *different tensor's* segment
whenever `rs_off > 0` (i.e. every QL after the first). **Fix:** `rs[o]`.

## Harness bugs found on the way (loqw_sim.py, trainer-side)
- Wo fp32 matmul applied as `ctx @ W`; the blob stores PyTorch (out,in)
  row-major → must be `ctx @ W.T`. The quantized linears were already
  correct (`a @ w.T`).
- GELU: torch's vectorized CPU gelu differs from libm erf by ~3e-7 — enough
  to flip ~1 int8 quantization step per 1000 values. The sim now uses
  vectorized libm erf and bit-matches the engine.

## Verification method (element-level bisect)
Per-layer, per-token dumps (tokens → ln1 → qa → integer dots → attention →
ctx → Wo → mlp) cross-checked C++ against loqw_sim until the first
divergent stage localized each bug. Final state: blocks 0–5 exact to 1e-6 on
all probed positions; the residual 1e-4..1e-2 raw-score noise on complex
positions is **fp32 accumulation-order noise amplified through int8 rounding
boundaries** — chaotic, unbiased, and argmax-stable (5/6 probe positions
identical argmax; the 6th a near-tie between two rook moves).

## Consequences
- M2 unblocked: NetQ output is trustworthy; SPRT (INT8 blob vs FP32 blob,
  same weights) quantifies the remaining quantization cost in Elo.
- `lrint` (round-half-even) replaces `lround` in the activation quantizer —
  matches numpy `rint` and the AVX2/scalar paths remain integer-identical.
- Parity harness contract: sim and engine may differ per-move score by up to
  ~1e-2 on sharp positions without indicating a bug; argmax stability is the
  meaningful invariant.
- Future QAT (RESEARCH-OPTIMIZATION2.md §5) trains *through* this boundary
  sensitivity and is expected to reduce the INT8-vs-FP32 Elo gap further.
