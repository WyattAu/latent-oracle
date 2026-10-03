# ADR-003: BC Played Moves as Primary Training Target

**Status:** Accepted (empirically proven, H1 SPRT + BC-v0 baseline)
**Date:** 2026-09-29

## Context

A searchless chess engine must select moves through pattern recognition
rather than tree search. The training signal determines what the model
learns to recognize. Three candidate training signals exist:

1. **BC (behavioral cloning):** predict the move a human actually played
2. **SF-label distillation:** predict the move Stockfish recommends
3. **Action-value prediction:** predict the outcome of each legal move

H1 tested #1 vs #2. ADR-002 documents the result: #2 is worse by ~309
Elo. This ADR formalizes #1 as the primary training target.

## Decision

Use **human played moves** as the primary training target. The model
learns: "in this position, what would a 2000+ rated player play?"

## Rationale

1. **Robust to imprecision:** the 2nd-best human move is usually
   acceptable. The 2nd-best SF move may be a blunder at 0-ply.
   Human moves are "forgiving" — playing a slightly different but
   still reasonable move doesn't lose the game.

2. **Pattern-matched to the inference model:** the model IS a pattern
   recognizer. Human moves ARE patterns (they arise from human thinking,
   which is pattern-based). SF moves arise from search, which is not
   pattern-based. Matching the data distribution to the inference
   mechanism is a fundamental ML principle.

3. **Proven empirically:** the BC model at 38-40% vs SF16-d1/d2 is
   the strongest configuration we've tested. All alternatives
   (SF distillation, Lc0 self-play at 3600 Elo) underperform BC.

4. **Scalable:** more Lichess months = more data, without new analysis
   infrastructure. The pipeline is proven (50M records sharded in
   minutes with the sidecar-based mask system).

## Consequences

- The model's ceiling is the average move quality of the training data
- Breaking through the ceiling requires either (a) higher-rated data,
  (b) quality filtering, or (c) a different training target (e.g., AV
  prediction). These are Phase B/C items.
- SF labels remain useful for VALUE head training (eval regression)
  but not for POLICY head training (move prediction)
