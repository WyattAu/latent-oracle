# Research Notes — RL Fine-Tuning of the Supervised Policy

**Scope:** the stage AFTER the BC → AV → quality chain: improving the policy
against a reward signal rather than more imitation. This is the direction the
2026 literature says is the biggest remaining lever at our scale.

---

## 1. The Faynt precedent (Oct 2026) — our exact scale

**Paper:** "Faynt: Scaling and Optimizing Policies for Competitive Melee"
- **10M- and 75M-param transformers** (we are 6.4M) pretrained on ~840K human
  replays, then RL-fine-tuned.
- The 10M model wins 98.4% vs 14 prior specialist releases after RL.
- Studies architecture, optimization, **hyperparameter transfer** from small
  to large pretraining — the same methodology as our HPO plan.

**Takeaway:** pretrain-on-human + RL is a validated recipe at 10M scale for
a combinatorial game with huge action spaces. Our pipeline already has the
pretrain half; the RL half is unbuilt.

---

## 2. QTPT: Q-targets beat imitation on weak data (Sep 2024→2026 line)

**Paper:** "From Weak Data to Strong Policy: Q-Targets Enable Provable
In-Context Reinforcement Learning"
- Behavior cloning on suboptimal trajectories is a **biased** learning
  signal; replacing BC with a Bellman-style Q-target objective makes the
  policy robust to data quality.
- This is the formal version of two of our results: (a) H1 negative —
  imitating SF-d16 labels hurt; (b) the decisive-weighting fix.

**Takeaway:** our planned AV-prediction phase IS a Q-target objective
(SF-eval of each move ≈ Q(s, a)). The theory says go further: train on
**Q(s,a) for SAMPLED policy moves**, not only the top-1 labeled move —
which is exactly the GRPO setup below.

---

## 3. GRPO for chess (no critic needed)

**Evidence:** "Fine-Tuning Autobidders with Group Relative Policy
Optimization" (Aug 2026) — GRPO replaces actor-critic for sequential
decision policies; DeepSeek-R1 lineage.

### 3.1 GRPO-for-chess design (concrete)
1. Sample a batch of positions from the BC/AV shard pool.
2. For each position, sample K = 16 moves from the current policy
   (temperature ≈ 0.8), filtered to legal.
3. **Reward:** SF16 eval of the child position (depth 12, mover POV),
   clipped to ±300cp, plus win/draw/loss if a rollout is used instead.
   Engine-free alternative: reward = value head's WDL of the child
   (self-referential; cheaper but risks drift).
4. Advantage A_i = (r_i − mean(r)) / (std(r) + ε) within the group of K.
5. Policy-gradient update with the importance ratio, clipped (PPO-style ε=0.2).
6. **KL anchor** to the frozen BC policy (β ≈ 0.02–0.05) — prevents mode
   collapse and legal-move distribution drift.

### 3.2 Why reward-from-SF-eval is the right first step
- It is "distillation from search" delivered through the RL gradient instead
  of hard labels — H1 showed hard-label imitation fails; RL with clipped,
  group-normalized rewards does not imitate, it *ranks*.
- The 5M labeled shard (in progress) already gives us the SF evaluator
  setup; the reward pass needs depth-12 single-position evals (~4ms each),
  so 16 moves × 512 positions ≈ 33s of SF time per training step — batch on
  the 6 CPU cores, overlap with GPU updates.

### 3.3 Prior art note
"When and Where to Trust the Teacher: Unifying On-Policy Distillation and
GRPO through Entropy-Calibration" (Sep 2026) formalizes the teacher-guided
GRPO hybrid — read before implementing §3; its entropy-calibration rule
likely replaces hand-tuning of the KL anchor.

### 3.4 Expected gain
Faynt: pretrain→RL ≫ pretrain alone. For us: +100–300 Elo plausible if the
BC policy is ~2000 and the reward signal is sound. Risk: reward hacking
(weird-but-SF-good positions) — mitigated by the KL anchor and by using
game outcomes for a later stage.

---

## 4. Regularized self-play with an anchored reference (Sep 2026)

**Paper:** "Steering Equilibrium Selection in Regularized Self-Play via the
Reference Policy" (DeepNash-family theory)
- Best-responding to a slowly moving, entropy-regularized reference policy
  converges toward Nash; anchoring the reference at a chosen policy steers
  WHICH equilibrium you land on.

### 4.1 Application
Stage 2 RL (after SF-reward GRPO works): self-play GRPO where the reference
policy is the frozen BC/AV net. Games between the current policy and the
anchor give outcomes as rewards — no SF in the loop at all. The anchor
prevents the classic self-play degeneracies (cycle formation, style collapse).

**Verdict:** PILOT P3 — after GRPO-from-SF-reward is validated.

---

## 5. What does NOT transfer

| Idea | Why not |
|---|---|
| PPO with a learned critic | Doubles model size; GRPO's group baseline is free |
| Reward = opponent-blunder rate | Rewards waiting for errors, not good play |
| Full AlphaZero self-play-from-scratch | 6.4M params + 1 GPU cannot bootstrap 2000+ Elo from zero; the human pretrain IS the boot |
| Offline RL (CQL/IQL) machinery | Our "offline" problem is really ranking with a queryable oracle (SF) — GRPO with oracle rewards is simpler and stronger |

---

## 6. Hyperparameters for the RL stage (start values)

| Param | Start | Search range |
|---|---|---|
| Group size K | 16 | 8–32 |
| Temperature (sampling) | 0.8 | 0.6–1.2 |
| KL anchor β | 0.03 | 0.01–0.1 |
| Clip ε | 0.2 | fixed |
| LR | 1e-5 | 5e-6–3e-5 |
| Reward clip | ±300cp | ±150–500 |
| Positions/step | 512 | 128–1024 |
| SF depth for reward | 12 | 10–16 |

Evaluation per checkpoint: 400-game fast SPRT vs the BC/AV base AND vs the
SF pool (the RL policy must beat both — beating only the base means drift).

## 7. Roadmap position

```
BC-v1 (running) → AV fine-tune (labels) → quality FT → RL-GRPO (this doc)
                                                        → self-play GRPO (anchored)
                                                        → DiffuSearch inference
```
The RL stage slots in BEFORE the DiffuSearch pilot: a stronger base policy
makes both the diffusion targets and the LoopCD contrasts sharper.
