# Research Notes: Endgame Ground Truth, Gumbel RL, Trajectory Consistency
## E1. Syzygy ground-truth training (ADOPT: highest ROI in the portfolio)
be rescored with EXACT game-theoretic values; no SF, no depth error, no
   exact-label positions. Still worth it; perfect labels exactly where
   the model is weakest; but it is a precision patch, not a data flood.
   min-|DTZ| toward the win / max toward the draw); a *perfect* policy
5. Later: extend to 6-man files (DTZ tables ~1-2 TB total; defer; 5-man
  *efficient* wins, not just winning; fewer 50-move/repertoire stumbles.
  tiny probe binary) OR; cheaper; do the rescoring in the RUST pipeline
  (`lo-data` already links Fathom? no; engine-side only). Pragmatic path:
## E2. Gumbel-GRPO: policy-improvement-guaranteed sampling for the RL stage
- the guarantee plain temperature sampling lacks.
  policy) and add a KL term toward it; the RL update then satisfies the
(TD(0)-style Bellman residual on static human games; Bellman residual
Bellman residuals are notoriously unstable at weight 1; use weight ≤0.1,
changing eval by 500cp... no; those are LEGITIMATE deltas; skip only