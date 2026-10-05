# Architecture — latent-oracle

Searchless chess engine: a 6.4M-param transformer picks the move in ONE
forward pass (no tree search); symbolic invariants (tablebases, repetition,
75-move, dead-position, low-clock PST) gate the neural choice.

## Components

```
src/
  position.hpp      PositionState (128B, two cache lines) + parse_fen/apply
  movegen/          attack tables + legal move generation (perft-suite gated)
  nn/
    net.hpp/.cpp    FP32 reference inference (LONW blob v1/v2/v3)
    netq.hpp/.cpp   INT8 VNNI inference (LOQW blob) — exact integer kernel,
                    parity contract with trainer/loqw_sim.py (ADR-005)
  engine_oracle/    move selection: net scores + symbolic gates + time mgmt
  uci/              UCI loop; magic-dispatch WeightsFile (LONW/LOQW)
  tb/               Syzygy probe via vendored Fathom (MIT)
tests/              perft suite + TB smoke (ctest)
```

## Contracts

- **Blob versions**: LONW/LOQW v1 ⊃ v2 (GAB) ⊃ v3 (castle/ep/king/rating/
  HiCo/bucketed value head). Trained by the latent-oracle-data repo.
- **Parity**: any inference change ships with harness evidence
  (FP32: ~4e-7 vs python; INT8: exact integer contract, argmax-stable).
- **Piece codes**: 1-6 white, 9-14 black (shard convention) — shared with
  the data pipeline's shards and the trainer's tokenizers.

## Research lineage

docs/RESEARCH-*.md (15 docs), docs/RESULTS.md (measured verdicts),
docs/adr/001-005, docs/SPEC-*.md. Provenance discipline: PROVENANCE.md.
