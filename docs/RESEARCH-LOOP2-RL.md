# Loop 2 — RL Formulation: what reward signal can a searchless net actually use?

**Question in the loop plan:** static-eval rewards were refuted (reward
hacking); outcome-based RL needs the right formulation at affordable scale.
What *is* the right formulation for a searchless, 6.5M-param, SF-distilled
engine?

**Answer, from our own measurements plus the literature:** at our scale there
is no formulation of RL that beats distillation from a fixed stronger oracle,
and the reason is structural rather than an implementation gap. This loop
converts that finding into (a) a spec-level ruling and (b) a permanent trainer
gate so the question cannot be re-litigated without new evidence.

## 1. What the literature actually requires of an RL formulation

| Line | Target the policy learns | Why the searchless case breaks |
|------|---------------------------|-------------------------------|
| AlphaZero / Gumbel AlphaZero | MCTS visit distribution + bootstrapped value | The target *is* the search. Removing MCTS removes the target; the network policy is then trained on itself. |
| Ceres (MCTS-free AZ, Inria 2024) | one-step TD(λ) actor-critic with gating + target networks | Works at 1000× fewer env steps on Atari, i.e. it removes the *search cost*, not the *need for a better-than-policy target*; still requires an asymmetric critic and careful anti-forgetting gating. |
| Self-imitation (our s4/s5) | winner moves of own games | Information-thin at small scale: measured `+2.32 ± 3.19` (10k games) and `−1.16 ± 3.93` (30k games). |
| DeepNoc / ChessFormer / Dragon | fixed strong oracle (SF search statistics), supervised | No reward to hack: the target cannot move. Divergence is bounded by the oracle. |
| Verifier-free RL (R-Zero-style, 2025) | model-as-its-own-verifier, bootstrapped | Chess results absent; same self-referential failure mode as our GRPO. |

The AlphaZero ablation is the load-bearing fact: the raw network policy
(equivalent to our engine's move selection, minus the value head) plays
hundreds of Elo below the searched policy. Our own SRTT numbers reproduce
this on our own nets: the AV head selected by argmax expected-win scored
**21.5% (−225 ± 38 Elo)** against the BC champion on 200 games, and the
raw-policy plays in every earlier test are likewise far below the champion.

## 2. Why our GRPO run failed — formulation, not bugs

The GRPO run was hardened until it was numerically sound (three real NaN
sources fixed: `nan_to_num` backward, log-ratio overflow, critic step-time
divergence; plus dead-engine recovery, resume, and non-finite guards). With
those fixed it trained for 1500 steps and produced:

- KL to reference → **~371 nats** (policy left the support entirely),
- reward → near-ceiling within ~300 steps,
- every checkpoint **loses** to its own teacher (best 48.5%).

That is textbook advantage collapse, not a bug: the critic is the *same
network* whose policy it is scoring, so the "advantage" is a re-derivation
of the current policy, and the cheapest way to raise it is to move the policy
into regions the critic has not been trained on. Group-relative
normalization removes the mean-shift but not the correlation. Any RL loop
that closes this loop (policy scored by its own or a co-trained head)
inherits the failure.

## 3. Consequence for the spec

**Ruling (applies to `SPEC-BLOB-V3.md` and the trainer):**

1. The blob carries no RL-specific artifacts — no value-iteration target, no
   temperature-sampling schedule, no critic. Move selection is a deterministic
   argmax over a single forward pass (the searchless contract).
2. Training is *supervised* by default. Any RL proposal must clear a gate
   before it is allowed to touch the engine.
3. **RL gate:** an RL run is admitted only if its final checkpoint beats *its
   own BC teacher* over 300 games at tc=15+0.2 (pair-model, engine-name
   recount), i.e. RL must add information beyond imitation. Our GRPO run
   failed this at every one of its 15 checkpoints, which is why it is
   refuted rather than tuned.

## 4. Where RL legitimately belongs (for the roadmap, not for this box)

- **Game generation** — RL/net policy for *data production* is where
  self-play pays off (LC0's 4B-game corpus). Our self-play loop is already
  operational at 37 min/10k games; the bottleneck is the number of games,
  not the formulation.
- **Large-scale outcome PG** (AlphaZero-style, MCTS or Ceres-style without)
  requires the 10²–10³× game volumes and the critic/trunk separation we do
  not have. On a 6-core box with 5.6 GB of VRAM this is a multi-GPU-month
  project, not an iteration.

## 5. Verification for this loop

- Build/perft/fuzz gates: unchanged, green (the ruling adds no engine code).
- Game-level: 200-game AV-vs-champion SPRT (loop 1) and the 800-game verdict-B
  confirmation are the evidence base; the 300-game teacher-relative gate is
  now written down as the admission criterion for any future RL work.

**Verdict:** the RL formulation question is settled *by measurement* for this
repo: distillation from a fixed oracle is the only formulation that produces
a gain here; self-referential RL is refuted at 15/15 checkpoints and 300-game
gates. Loop 2 closes with a spec ruling + gate, not a new training run.
