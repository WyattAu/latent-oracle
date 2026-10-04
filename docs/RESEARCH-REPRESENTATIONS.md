# Research Notes — Representations, Values, Alternative Trunks

**Scope:** upgrades to WHAT the network learns (auxiliary structure, value
geometry) and HOW (trunk families). Companion to RESEARCH-RL.md.

---

## 1. Distributional value geometry

### 1.1 Current state
Our value head outputs W/D/L softmax — already a 3-bin distributional head
(better than scalar regression; SF's WDL logistic target matches).

### 1.2 Upgrades ranked
1. **Expectile/quantile heads** (Implicit QL line): predict τ-quantiles of
   the outcome distribution instead of the mean. Gives risk-aware move
   selection (engine could prefer high-upside moves when losing). Cost: +2
   output dims. Verdict: **PILOT P3** — needs the RL stage first to have a
   use for risk sensitivity.
2. **Finer bins** (C51-style, 51 bins of eval): richer value signal for the
   AV phase; the draw-gate logic could read P(eval>0) directly. Cost: head
   reshape + target binning (we HAVE SF evals). Verdict: **ADOPT with the
   AV trainer** if per-move evals land in shard format v2.
3. **Two-hot eval encoding** (Lc0 practice): value target = 2-point
   interpolation of eval on a bin grid — smoother than CE on a single bin.
   Cheap, proven. Verdict: **ADOPT with AV phase.**

---

## 2. Auxiliary self-supervised tasks (representation shaping)

Multi-task auxiliary losses improve main-task representations when labels
are free (they are — the shard has everything needed):

| Task | Target | Cost | Expected |
|---|---|---|---|
| **Next-position prediction (JEPA-lite)** | latent(s_{t+1}) from s_t + a_t; cosine or CE on quantized latents | small head | +20–50 Elo (representation quality) |
| **Opponent-reply prediction** | predict the move actually played next by the opponent | reuses policy head | +10–30 |
| **Eval-delta prediction** | predict sign/size of eval change after the played move | 3-bin head | +10–20 |
| Piece-count / material regression | trivial | ~0 | marginal |
| **Move-legality probe** | predict legality of random (from,to) pairs | small head | diagnostic value; test-time legality gate |

Implementation: heads read the trunk output; losses weighted 0.1–0.3 each.
The JEPA-lite task is the most interesting — it forces the trunk to encode
dynamics (what Lc0's look-ahead interpretability found: strong nets
internally represent future states). Verdict: **ADOPT JEPA-lite + reply
prediction in the AV trainer**; keep the rest behind flags.

---

## 3. Trunk families (beyond transformer)

| Trunk | Evidence | Verdict for us |
|---|---|---|
| Transformer (current) | everything above | KEEP |
| **Mamba-3 / SSM** | 2026 state-tracking work; linear-time sequence mixing | REJECT for now: our sequences are 64-67 tokens (parallelism moot); bidirectional attention is REQUIRED by the diffusion path; GAB already injects chess geometry |
| xLSTM | similar | same rejection |
| Graph attention (pieces as nodes) | AMCTS-adjacent 2026 work; edges = attack relations | WATCH — natural fit for chess but a rewrite; revisit only if GAB under-delivers |
| CNN (Lc0-style ResNet) | proven at scale | REJECT — transformer already competitive at our size |

---

## 4. Latent world models / value-equivalence (WATCH list)

- "Operator-on-F complements value-equivalence: a planning-time diagnostic
  for latent world models" (Jul 2025): diagnostics for whether a latent
  model's rollouts preserve values.
- JEPA-TTT (Sep 2026, already in RESEARCH-INFERENCE.md §4).

These matter only if we build an explicit latent dynamics model. The
DiffuSearch pilot IS a lightweight dynamics model over FEN tokens — its
denoising quality on s1/s2 tokens will tell us whether deeper latent-planning
investment is warranted. **Decision gate:** if DiffuSearch a0-match ≥ 40%
(§6 of SPEC-DIFFUSION), fund a latent world-model study; else skip.

---

## 5. Interpretability-informed training tweaks

From the chess-transformer interpretability line (Tracing-the-Thought 2026,
look-ahead probes):
- Strong nets internally predict the OPPONENT'S best reply — our reply-
  prediction auxiliary (§2) aligns training with what the architecture
  wants to learn anyway.
- Move-matching saturates while strength grows — do not over-tune on
  move-matching (see RESEARCH-EVAL.md §3.1 caveats); the puzzle suite is
  the better triage metric.
- superposition findings suggest wider d at the same param count may help
  feature clarity: if we ever rebuild, prefer d=320/6L over d=256/8L at
  equal params. Verdict: note for v1.0 architecture review, not now.

---

## 6. Verdict table

| Technique | Cost | Expected | Verdict |
|---|---|---|---|
| Two-hot eval value targets | small | better value learning | ADOPT (AV phase) |
| JEPA-lite next-position aux | small head | +20–50 Elo | ADOPT (AV phase) |
| Reply-prediction aux | ~free | +10–30 Elo | ADOPT (AV phase) |
| Quantile value head | small | risk-aware play | PILOT P3 (post-RL) |
| Legality probe head | tiny | diagnostic | ADOPT (free signal) |
| Mamba/xLSTM trunk | rewrite | unclear at 64 tokens | REJECT |
| Graph-attention trunk | rewrite | promising | WATCH |
| Latent world model | large | unknown | Decision gate = DiffuSearch a0-match |
