# Loop 8 — Endgame: runtime Syzygy exists, now measured

**Question in the loop plan:** TB sidecar null at training; runtime TB paths
in `src/tb` unexplored.

## State of the code (found, not built this loop)

- `src/tb/tb.{hpp,cpp}`: Fathom-based (MIT) Syzygy **root probing**, active
  at ≤5 men with no castling rights; DTZ/WDL-optimal move overrides the
  policy. The header names the design rationale: this converts won endgames
  provably — the exact blind spot of a searchless policy.
- Wired into UCI (`SyzygyPath` setoption → `tb::init`).
- **No SPRT verdict in this program ever set SyzygyPath** — every match in
  RESULTS ran TB-off on both sides: consistent A/B, verdicts unaffected.

## Measurement (this loop)

TB sidecar at TRAINING was measured null long ago; the open question was the
RUNTIME path. Match: new champion `net_q_v2` TB-on (145 files, 3-4-5 men,
930 MB at /home/wyatt/data/syzygy) vs TB-off, 300 games tc=15+0.2.
Expectation: small positive (the 2000-line book rarely reaches ≤5 men, and
games that do are often already drawn), but the win-conversion guarantee is
valuable beyond Elo (searchless engines resign/hallmark badly in won
endgames).

**RESULT (2026-10-21): perfect null.** 300 games, 150/150 pairs split —
the TB path never changed a single game outcome under the 2000-line book at
tc=15+0.2 (the book's openings effectively never reach a ≤5-men position
where TB would decide differently). Combined with the training-side TB
sidecar null, the endgame levers are measured harmless-and-useless *in this
harness*. The code stays: free insurance for real deployments that reach
≤5 men, and the only provably-correct play path in the engine.

**Loop 8 verdict:** runtime TB implemented, measured no-op in harness,
retained as a correctness guarantee. No strength lever found.
