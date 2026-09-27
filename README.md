# latent-oracle

A searchless neural chess research engine. Strictly 0-ply: one forward pass
from position to move, augmented only by O(1) symbolic invariants
(legality, repetition, the 75-move rule, dead positions).

**Thesis:** how strong can a searchless policy play chess as a function of
parameters, label quality, and spatial inductive bias — measured under SPRT?
DeepMind answered one point on this curve at 270M parameters
(*Grandmaster-Level Chess Without Search*, 2024); this project maps the
small-budget frontier of the same curve.

Apache-2.0, clean-room ([provenance](docs/PROVENANCE.md)). Stockfish is used
strictly as a binary label generator in training (M1+); no GPL source is read.

## Status: M0 — board core

| Milestone | Scope | State |
|-----------|-------|-------|
| M0 | Board core: 128-byte `PositionState`, PEXT/magic/classical movegen, perft to d6, UCI, symbolic draw gates | **done** |
| M1 | Lichess DB + Stockfish-labeled distillation; first policy-only Elo | **first numbers live** (see below) |

## First results (research map, v0)

| Engine | Conditions | Result |
|--------|-----------|--------|
| BC-v0 (6.4M params, 2 epochs × 20M Lichess positions) | SPRT 200 games vs SF16 `go depth 1`, openings from own shard | **51.75% (74W/59D/75L) ≈ SF16-d1 level** |

Provenance: `sprt/bc-v0.pgn` + phase-C logs; opponent pool SF16.1 fixed
depth, 60+6 clock, 2000-position EPD openings sampled from the training
shard. Distilled (1M/5M SF-label) numbers pending.
| M2 | INT8 AVX-512 VNNI inference; dot-product policy head; Engine A complete | — |
| M3 | Architecture ablations (the research map) | — |
| M4 | Engine B: micro-eval quiescence verification layer, node-budget curve | — |

## Build

```sh
cmake --preset debug && cmake --build --preset debug -j
ctest --preset debug
```

Requires CMake 3.28+, a compiler with C++26 support (degrades to C++23/C++20
automatically), and Ninja. x86-64: BMI2 is used when present at runtime,
with a magic-bitboard fallback otherwise; no `-march` flag is required for a
correct binary.

## Usage

```sh
./build/debug/latent-oracle                 # UCI engine
./build/debug/latent-oracle perft 5 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" --divide
./build/debug/latent-oracle perftsuite      # full CPW suite to d6/d5
./build/debug/latent-oracle perftsuite --quick
```

## Tablebases

Syzygy 3-4-5-man WDL/DTZ tables are supported via vendored Fathom: when a
position has ≤5 men and no castling rights, the engine plays the DTZ-optimal
tablebase move (provably converting won endgames — the classic blind spot of
searchless policies). Point it at a table directory with:

```
setoption name SyzygyPath value /path/to/syzygy
```

Tables: https://tablebase.lichess.org/tables/standard/3-4-5-*

## Design notes

* `PositionState` is exactly 128 bytes (two cache lines): 12 piece bitboards,
  incremental Zobrist key, packed state. Occupancy is derived, not stored —
  this closes the layout math the original spec got wrong (137 bytes).
* Deterministic Zobrist keys (consteval splitmix64): identical across builds.
* En passant targets are normalized to *capturable-only*, keeping repetition
  detection exact.
* The M0 engine gates draws correctly: a winning position will never step
  into three-fold repetition or cross the 75-move line; a losing one will
  seek the draw.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[PROVENANCE](docs/PROVENANCE.md).
