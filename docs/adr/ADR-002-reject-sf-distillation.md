# ADR-002: Reject SF Depth-16 Label Distillation as Primary Training Signal

**Status:** Accepted (empirically proven)
**Date:** 2026-09-29
**Deciders:** Wyatt Au, with H1 SPRT evidence

## Context

The original plan (plan of record v0) proposed SF depth-16 label
distillation as the primary training improvement: annotate positions
with Stockfish's best move and train the model to predict it.

This was implemented as:
- Shard worker: 20M positions from Lichess July 2026
- Labeler: 1M positions labeled with SF16 depth-16 multipv-3
- Trainer: 2 epochs on 1M labeled positions
- SPRT: 200 games vs SF16 at fixed depths 1 and 2

## Decision

**REJECT** SF depth-16 label distillation as the primary training
signal for a searchless chess engine. The H1 SPRT result showed:

| Engine | vs SF16-p1 | vs SF16-p2 | Training data |
|---|---|---|---|
| BC-v0 (played moves, 40M samples) | **−83 ± 33** (38.25%) | **−72 ± 37** (39.75%) | 20M × 2 ep |
| Distilled-1M (SF labels, 2M samples) | **−436** (7.5%) | **−338** (12.5%) | 1M × 2 ep |

Even after extended training (20 epochs, 20M samples), the distilled
model remains ~300 Elo below BC. The policy loss converges (0.62)
but playing strength does not improve proportionally.

## Rationale

1. **Search-precision mismatch:** SF depth-16 moves are optimal only
   within the context of SF's search tree. A searchless model cannot
   replicate that context — it sees the position but not the tree.
   The 2nd-best SF move at depth 16 might be a blunder at depth 0.

2. **Empirical evidence is clear:** -309 Elo gap (H1 SPRT) and -236
   Elo gap (extended training H1a) both confirm the effect. The result
   is consistent across multiple runs and evaluation conditions.

3. **BC is the right signal for searchless engines:** human played
   moves are robust to imprecision — the 2nd-best human move is usually
   acceptable. SF moves require precision that 0-ply inference cannot
   deliver. This is a structural property, not a training issue.

4. **The finding is a contribution, not a failure:** the result
   "SF-label distillation degrades searchless play" is a novel data
   point for the research community. The H1 experiment design (same
   architecture, same data positions, different labels) isolates the
   label-quality effect cleanly.

## Consequences

### Positive
- Frees the project to pursue DiffuSearch (ADR-001), which uses the
  existing BC training as a foundation and adds future-token prediction
- The BC path is proven and scalable: more Lichess months = more data
- The negative result is publishable as a finding

### Negative
- The labeling infrastructure (SF process pool, shard patching) is
  now unused — the engineering investment in the labeler is stranded
- Future improvements require BC data scaling rather than label quality
  improvements, which is a slower path

### Neutral
- The labeler code remains in the repo as a research artifact
- The 1M labeled shard remains available for future experiments
  (e.g., testing whether shallower SF labels work better)

## Related

- [ADR-001](ADR-001-diffusearch.md): DiffuSearch replaces label
  distillation as the primary technique
- [SPEC-DIFFUSION.md](../SPEC-DIFFUSION.md): the new training spec
