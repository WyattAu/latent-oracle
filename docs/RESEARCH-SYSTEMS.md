# Research Notes — Systems: Performance, Latency, Robustness, Rigor

**Scope:** engineering-quality research. Capability research (rounds 1–6)
raises Elo; this doc makes sure the numbers are honest, the engine is fast
enough to test at volume, and nothing breaks under stress.

---

## 1. Measured performance (2026-10-04, i5-9400F, loaded box)

| Path | Median eval¹ | Throughput | Notes |
|---|---|---|---|
| FP32 (`nnpar`) | 2914 ms | ~9 GMAC/s effective | scalar/autovectorized |
| **INT8 (`nnqpar`)** | **616 ms** | ~0.6 GMAC/s effective | **4.7× speedup — VNNI validated** |
| process spawn overhead | 4.8 ms | — | negligible |

¹ Includes FEN parse + legal movegen + score printing; contention from 7
labeler SF processes + GPU training active during measurement.

### 1.1 Findings
1. **INT8 4.7× is real** and directly converts to SPRT throughput (more
   games per concurrency-hour).
2. **Both paths are far from theoretical throughput.** INT8 should hit
   10–50 GMAC/s single-thread (AVX2 madd path is ~350M MACs/eval);
   616 ms implies ~0.6. Expected eval time ~30–80 ms. Suspects (in order):
   - system contention (7 SF procs saturate all cores),
   - per-element `lrint` in the activation quantizer (1024/step, 2560
     calls/eval = ~2.6M scalar roundings),
   - activation vector re-read per output row (cache-friendly but still
     4096 redundant loads per QL),
   - denormals from the fp32 dequant path (no FTZ/DAZ flags set).
3. **Latency budget check:** at 616 ms/move, a 15+0.2 blitz game (~40 moves)
   spends ~25 s/move-side of thinking — viable but tight under
   concurrency. A 5× speedup would double SPRT throughput for free.

### 1.2 Optimization backlog (ordered by expected ROI)
| Item | Est. gain | Cost |
|---|---|---|
| Batch the quantizer: vectorized `lrint` with AVX2 (v cvtps2dq) | 2× | ~30 lines |
| Precompute `(128 − zp)·rs` outside the o-loop | small | trivial |
| Reorder loops: row-block tiling for L2 reuse of q_a | 1.3–1.7× | medium |
| `_mm256_setcsr()` FTZ/DAZ on | guards denormal stalls | 2 lines |
| Thread the o-loop over 2 threads (SPRT uses concurrency 5; per-engine
  threading trades against fastchess concurrency — measure both ways) | ≤2× | small |
| Spawn-free self-play mode (single process, in-process evals) | removes
  5 ms + page-cache warmth | medium |
| **Item 0: re-measure on an idle box** before any of this — contention may
  explain most of the gap. | — | — |

---

## 2. Robustness inventory

### 2.1 Covered (verified by tests or SPRT history)
| Failure mode | Guard |
|---|---|
| Illegal move | argmax over legal-move mask only |
| Repetition draws given away | repetition gate in `bestmove_net_impl` (win→avoid, lose→seek) |
| 75-move rule | halfmove gate |
| Dead positions | insufficient-material check |
| Lost endgames | Syzygy DTZ probe ≤5 pieces (Fathom, tested) |
| Time forfeit under 10 s | PST fallback (µs-class) |
| Corrupt shard records | trainer-side legality re-check (mask/target match) |
| fp16 training overflow | GradScaler + clip 1.0 |
| Net file wrong/absent | graceful PST fallback + info string |
| INT8 numeric drift | parity harness (exact integer contract, ADR-005) |

### 2.2 Gaps (backlog)
| Gap | Risk | Fix |
|---|---|---|
| No opening book | move-1 noise, SPRT opening leakage (fixed 2000 EPD) | 1-ply book from shard (RESEARCH-DATA §5) |
| No ponder/contempt | product polish only | v1.x |
| PGN round-trip untested | historical record validity | test: play 50 self-games, reparse with python-chess |
| GAB blob v2 + NetQ | INT8 path rejects v2 blobs (by design; v2 runs FP32) | NetQ v2 support after QAT |
| Time management naive (fixed PST threshold) | weak clock play at fast TCs | log-based TC adaptation (future) |
| No crash-recovery in long self-play | lost games on crash | checkpoint game batches |

---

## 3. Statistical rigor

| Practice | Status |
|---|---|
| SPRT with pre-set bounds, PGNs archived | ✓ every verdict |
| Power analysis (400 games resolve ≥30 Elo; 1200 for close calls) | ✓ RESEARCH-EVAL |
| Keep-best gating (epoch exports SPRTed, winner promoted) | ✓ `keep_best.sh` armed |
| Paired openings (same EPD pool both sides, color pairing) | ✓ fastchess `-repeat` |
| Seed control (trainers seed 0; eval slice fixed) | ✓ |
| Pre-registered kill criteria (H2, AMZ ladder, N1/N2) | ✓ in research docs |
| **Pentanomial score model** (pair-level statistics) — fastchess reports
  trinomial; pair-aware CIs are tighter | TODO: switch verdict lines to
  pentanomial pairs (report W/D/L per game-pair) |
| Elo floors: differences <30 Elo treated as ties | ✓ policy |

---

## 4. Testing rigor

| Layer | Tests | CI |
|---|---|---|
| Engine movegen | perft suite (full in CI, release-grade) | ✓ engine repo |
| Tablebases | TB smoke (KQvK mate-in-1) | ✓ |
| FP32 inference parity | `export_parity.py` (5e-6) | manual — TODO: CI-ify with tiny random net |
| INT8 inference parity | loqw_sim ↔ `nnqpar` exact-integer contract | manual (ADR-005) — TODO: CI with tiny random blob |
| Trainer stack | NEW: 7 tests — shard round-trip, GAB identity, recycle
  grads, Muon, RCT finiteness, diffusion tokenizer | ✓ data repo (this commit) |
| Diffusion pipeline | smoke (build→train→denoise) | manual |
| **Gap: movegen fuzzing** | perft covers structured positions; random
  FEN fuzz (10k random legal positions, compare move counts vs python-chess)
  would close the last movegen risk | TODO small |

---

## 4.5 Sample-slice validation policy (added 2026-10-05)

**Standing rule: every bulk dataset build validates a small slice before the
full scan.** After the diffusion tokenizer poisoning (black pieces shifted
×2) wasted a 2M-sample build, `build_samples`, `amz_pilot`, and
`make_tb_labels` now fail fast at ~50–2000 items on:
- token/target range and shape invariants,
- decode round-trip piece sanity (exactly one king per side),
- one real forward/backward step (finite loss).

Negative test: re-injecting the original bug trips the gate at the
king-count assertion. Cost when clean: seconds. Cost when dirty: minutes
instead of hours. This generalizes the fuzz-test lesson (fuzz_movegen
caught the ep bug) to dataset construction.

## 5. Conclusions

1. The INT8 path's 4.7× is production-real; a further ~5× is on the table
   (quantizer vectorization + idle-box re-measurement) and converts directly
   into SPRT volume.
2. Robustness coverage is strong where it matters (legality, draws, TB,
   clocks); the honest gaps are books, PGN round-trip testing, and v2-blob
   INT8 support.
3. Rigor: the data repo now has CI (7 tests) closing the biggest gap; the
   remaining TODOs are CI-ifying the two parity harnesses and pentanomial
   verdict reporting.
