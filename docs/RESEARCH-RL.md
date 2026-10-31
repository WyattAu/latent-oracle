# Research Notes: RL Fine-Tuning of the Supervised Policy
## 1. The Faynt precedent (Oct 2026): our exact scale
  to large pretraining; the same methodology as our HPO plan.
- This is the formal version of two of our results: (a) H1 negative -
**Q(s,a) for SAMPLED policy moves**, not only the top-1 labeled move -
Optimization" (Aug 2026); GRPO replaces actor-critic for sequential
6. **KL anchor** to the frozen BC policy (β ≈ 0.02–0.05); prevents mode
  of hard labels; H1 showed hard-label imitation fails; RL with clipped,
  so 16 moves × 512 positions ≈ 33s of SF time per training step; batch on
GRPO hybrid; read before implementing §3; its entropy-calibration rule
(weird-but-SF-good positions); mitigated by the KL anchor and by using
anchor give outcomes as rewards; no SF in the loop at all. The anchor
**Verdict:** PILOT P3; after GRPO-from-SF-reward is validated.
| Offline RL (CQL/IQL) machinery | Our "offline" problem is really ranking with a queryable oracle (SF); GRPO with oracle rewards is simpler and stronger |
SF pool (the RL policy must beat both; beating only the base means drift).