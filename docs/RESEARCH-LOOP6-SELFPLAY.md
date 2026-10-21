# Loop 6 — Self-play generation: diversity is book-capped and replay is deterministic

**Question in the loop plan:** the loop is proven (37 min / 10k games);
opening diversity and game-quality filters unexplored.

## Measurement (2026-10-21, on the existing corpora)

| Corpus | Games | Distinct first moves | Distinct ply-8 positions | Dup rate ply-8 |
|---|---|---|---|---|
| s1 | 10,000 | 850 (book-seeded FENs) | 1,982 | 80.2% |
| s4 | 20,000 | 850 | 1,970 | 90.1% |

The generator seeds games from `openings.epd` — exactly **2,000 lines** —
and plays each deterministically (argmax). Measured consequence: the
30k-game s4/s5 training corpora contain only ~2,000 distinct opening
states, each replayed 10-15× near-identically. The data was
**state-space-thin**, compounding the outcome-signal thinness that made
s4/s5 nulls (winner-move BC at 74% draws).

## What other engines do

- **LC0**: billions of games; root Dirichlet noise + temperature sampling
  during generation; positions consumed raw (no dedup needed at their scale).
- **AlphaZero**: root Dirichlet (α=0.3 chess) + temperature 1.0 for 30 plies,
  then argmax; opening selection by sampling the most recent ply-1 policy.
- **DeepNoc**: not self-play for strength (distillation), but their 9M GM-game
  dataset has millions of distinct seeded positions.

## Verdict

Diversity is **book-capped at 2,000 seeds** and replay is deterministic:
two separate one-line fixes exist —

1. seed from a larger pool (the SPRT book is a measurement tool, not a
   generation corpus; LC0-style: sample random plies from prior games as
   seeds, unbounded),
2. root temperature 1.0 for the first 8-12 plies (deterministic argmax after),

with the gate: **<10% ply-8 duplicate rate per 10k games**.

But the honest payoff analysis: fixing diversity only pays when paired with
a data-hungry training target. Outcome-move BC is information-thin at ANY
diversity (s4/s5: 30k games → −1.16 ± 3.93), and the AV distillation target
is budget-bound (loop 4). So the fix is recorded as spec + gate, deferred
until a training target exists that can consume the extra entropy.

**Loop 6 verdict:** measured and closed — diversity is book-capped
(~1,980 distinct early positions across 30k games, 80-90% duplicates);
generator fix specified (unbounded seeding + root temperature) with a
quantitative gate; deferred behind a viable training target.
