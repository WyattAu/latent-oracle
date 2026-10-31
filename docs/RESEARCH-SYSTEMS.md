# Research Notes: Systems: Performance, Latency, Robustness, Rigor
| **INT8 (`nnqpar`)** | **616 ms** | ~0.6 GMAC/s effective | **4.7× speedup; VNNI validated** |
| process spawn overhead | 4.8 ms | - | negligible |
   spends ~25 s/move-side of thinking; viable but tight under
  threading trades against fastchess concurrency; measure both ways) | ≤2× | small |
| **Item 0: re-measure on an idle box** before any of this; contention may
  explain most of the gap. | - | - |
| **Pentanomial score model** (pair-level statistics); fastchess reports
| FP32 inference parity | `export_parity.py` (5e-6) | manual; TODO: CI-ify with tiny random net |
| INT8 inference parity | loqw_sim ↔ `nnqpar` exact-integer contract | manual (ADR-005); TODO: CI with tiny random blob |
| Trainer stack | NEW: 7 tests; shard round-trip, GAB identity, recycle