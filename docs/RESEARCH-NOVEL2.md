# Research Notes — Novel Techniques II: Toward a Genuine Breakthrough

**Scope:** two new designs attacking the core limitation — a searchless net
cannot calculate. Round-5 techniques (HiCo, RCT) improved the net's
*inputs* and *computation shape*; these attack its *training targets* and
*forward graph*: the two places where a qualitative jump is still possible.
Prior-art checks logged honestly at the end.

---

## N4. CTAP — Child-Token Attention Policy

### Idea
AV prediction (Ruoss et al.) scores each child with a SCALAR value and
argmaxes. But comparing children is a *relational* computation: "this
capture wins a pawn but opens my king" needs both child representations
simultaneously. Scalars throw that away.

**Design:** expand the forward pass itself.
1. For each legal move m (~30), compute a child-summary embedding: run the
   child position through the first L/2 trunk layers, mean-pool → d-dim.
   All children share the batch dimension — one extra half-depth stack.
2. Append ~30 child tokens to the current position's sequence (each tagged
   with its move-id embedding).
3. Run the FULL trunk over [position tokens ‖ child tokens]; the policy
   head reads the position tokens, now attending over what every move
   actually leads to — representationally, not as scalars.

This is **amortized 1-ply search inside the forward graph**: the net sees
every option the way a 1-ply search would, but the "search" is one batched
matrix stack and the comparison circuitry is learned. Subsumes AV (the
trunk can implement scalar ranking internally — but can also detect that
two candidate captures fork the queen). Composes with recycling (later
passes refine the comparison) and with LoopCD.

### Implementation map
- Children from shard pairs: the diffusion pilot's `find_connecting_move`
  machinery, extended to enumerate ALL legal children (python-chess apply;
  ~30 per position, dataloader-side).
- Zero-init gate on child tokens → warm-start from any existing checkpoint
  bit-exactly (the GAB trick, third use).
- Engine: children generated with the existing movegen; child embeddings
  share weights. Blob v3+ (needs a second tensor group for the child
  encoder layers — L/2 copies of the trunk weights, OR reuse trunk weights
  for children (weight-sharing, cheaper, try first).
- Cost: ~1.5–2× single-pass inference; ~2× training step.

### Test and kill criteria
- Pilot: fine-tune BC-v1 → CTAP on 500k paired positions; puzzle suite must
  beat the AV baseline by ≥2 points (paired slice); SPRT vs SF-p1.
- Mechanism check: CTAP policy agreement with SF's top-3 ranked children
  should EXCEED the AV-scalar agreement (relational info has value) — else
  the trunk is just re-deriving scalars and the complexity is waste.
- Kill: no puzzle gain, or engine latency blowup beyond 2×.

---

## N5. AMZ — Amortized Minimax (the oracle-ceiling breaker)

### The ceiling we're stuck under
Every target in our pipeline ultimately derives from SF: BC labels (human),
AV labels (SF-d14), quality filters (SF). The policy can never reliably
exceed the oracle — H1 showed the precision mismatch; and SF labels are
expensive (~3 days per 5M). GRPO with SF rewards (RESEARCH-RL) stays under
the same ceiling.

### Idea
Compute policy targets with the net's OWN value head inside a 2-ply minimax
over the LOSS, bootstrapped across epochs:

```
Q̂(s, m) = min_{r ∈ Replies(s,m)} V̂(child(child(s, m), r))
π-target(s) = softmax(β · Q̂)          # new policy target
L = CE(policy(s), π-target) + standard value losses
```

- `Replies(s,m)`: K=6 replies sampled from the reply-prediction head
  (realistic opponent moves, not uniform), plus the opponent's actual game
  move when known.
- `V̂` = the net's own value head on the grandchild position.
- **Oracle annealing:** `Q̂ = α·SF_eval + (1−α)·minimax-own`, α: 1 → 0.2
  across training epochs. Early: SF shapes the value head. Late: the net's
  own 2-ply judgment drives the policy. Targets refresh each epoch
  (bootstrap) — as V̂ improves, targets sharpen, a self-improvement loop
  with NO search and NO oracle at the limit.

### Why this is breakthrough-shaped
1. **It bakes calculation into weights.** "Consider their best reply" is
   exactly what weak play lacks; AMZ forces every training position to be
   evaluated through a 2-ply tree at LOSS time, so the trained policy
   encodes two-ply tactical awareness as pattern recognition — the
   calculation happens once, during training, amortized forever.
2. **It breaks the SF ceiling** — the only mechanism in our portfolio that
   does. DiffuSearch/RL/AV all consume external signal; AMZ's signal
   improves with the net.
3. Cheap: K=6 replies × ~30 moves ≈ 180 value-head evals per position —
   batched, ~2-3× training cost, zero inference cost (policy stays one
   forward pass).

### Honest risks
- Pessimism bias: min-over-K underestimates Q when K misses the best
  reply... (actually overestimates safety — min over a SUBSET is optimistic
  wrt the true min. Either way biased.) Mitigate: sample replies from the
  learned reply distribution (high-probability replies only), and mix
  oracle signal via the annealing floor.
- Self-distillation collapse: targets from own weights can amplify errors.
  Mitigate: KL anchor to the BC policy, α-floor, EMA teacher as the target
  V̂ (mean-teacher stabilization).
- The value head must be good enough to start (needs the AV phase first).

### Experiment ladder (pre-registered)
1. **Offline pilot:** with BC-v1's value head, compute AMZ targets for
   100k positions; 1 fine-tune epoch; puzzle suite vs BC-v1. Gate: ≥ +1.5
   puzzle points.
2. **Bootstrap check:** 3 target-refresh iterations. Gate: monotone Elo
   increase per iteration on fast SPRT (any 2 of 3 iterations positive).
3. **Annealing run:** α 1→0.2 over 3 epochs from the AV net; official SPRT
   vs SF-p1/p2 and vs the AV net.
- Kill criteria: iteration-2 Elo < iteration-1, or collapse (draw rate
  explosion / move-diversity crash).

### Differentiation
Worst-case/robust training exists generically; differentiable game solving
exists (non-chess). Loss-side minimax over OWN heads, legality-masked,
oracle-annealed, on a static human dataset, for searchless chess — no
prior art found (checked: amortized minimax, implicit-search targets,
oracle-free policy improvement; closest neighbors are Offline Fictitious
Self-Play 2024 and Expert Iteration — both still search/solver-bound).

---

## N6. LID — Latent Iterative Deepening (spec, post-DiffuSearch)

DiffuSearch denoises ONE canonical future. LID: recycling passes manage M
latent branch slots — pass 1 ranks branches, pass 2 prunes weak ones and
expands survivors in latent space, pass k deepens — MCTS as a fixed
computational graph over learned latents. Deferred until DiffuSearch
verdicts land: if a0-match ≥ 40% the latent machinery works and LID is
the natural escalation; if not, latent branches inherit the same failure.

---

## Build order

| Step | Item | Depends on | Cost |
|---|---|---|---|
| 1 | N5 offline pilot (100k targets, 1 epoch) | BC-v1 weights (have), value head quality | hours |
| 2 | N4 pilot (500k pairs, zero-init gate) | blob v3 data path | 1 day |
| 3 | N5 bootstrap iterations ×3 | pilot-positive | 2 days |
| 4 | N5 annealing run + official SPRT | bootstrap-positive | 1 day |
| 5 | N6 spec | DiffuSearch a0-match ≥ 40% | — |
