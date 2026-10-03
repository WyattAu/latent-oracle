# Implementation Roadmap — Full Evaluation

**Date:** 2026-10-03
**Status:** Final — reviewed against all ADRs, specs, and empirical results

---

## Executive Summary

Latent-Oracle is a searchless chess research engine. The thesis: how
strong can a chess engine play with zero tree search, using only
pattern recognition through a neural network? We answer this by
mapping the design space across data source, training target, model
size, and inference technique — measuring each dimension's contribution
through controlled SPRT experiments.

**Current result:** 38–40% vs SF16 depth 1/2 (~2000 Elo equivalent).
**Target:** 2400+ Elo through the techniques identified in this roadmap.

---

## Phase A: Foundation (COMPLETE ✅)

| Deliverable | Proof |
|---|---|
| Board core: 128B PositionState, PEXT/magic/classical movegen | Perft d6 CPW suite green |
| Legal movegen with EP pins, promotions, castling | 119M+ nodes verified |
| Syzygy 3-4-5-man root probe | KQvK mate-in-1 verified |
| Time management: low-clock PST fallback | No forfeits in 200-game SPRT |
| UCI: full protocol, SPSC ring IO thread | 200-game fastchess match clean |
| v0.1 released | GitHub tag + release notes |

---

## Phase B: H1 Ablation (COMPLETE ✅)

**Question answered:** Does SF depth-16 label distillation improve a searchless engine?

**Answer: NO.** SF-label distillation degrades play by ~309 Elo vs BC.

| Experiment | Result |
|---|---|
| BC-v0: 20M played moves, 2ep | 38–40% vs SF16-d1/d2 |
| Distilled-1M: SF labels, 2ep | 12–14% vs SF16-d1/d2 |
| Verdict | **H1 NEGATIVE — BC wins by ~300 Elo** |

**Implication:** SF depth-16 moves require search precision. A searchless
model cannot replicate that precision. BC moves are forgiving — the
model can approximate them without catastrophic results.

**Documented in:** ADR-002, README results table

---

## Phase C: BC Scaling + Data Source Ablation (RUNNING ⏳ + QUEUED)

### C1: BC-v1 (50M human positions, 3 epochs)
- **Status:** RUNNING (epoch 0, step ~11k/97k)
- **Expected:** +50–100 Elo over BC-v0 (2.5× more data)
- **SPRT after training** → v0.2 if improved

### C2: Lc0 self-play (3M positions from 3600 Elo engine)
- **Status:** READY (shard built, 54K games)
- **Purpose:** data-source test (bot-generated vs human-generated)
- **Training:** queued after C1 (GPU contention)

### C3: Quality filtering
- **Method:** combined filter (SF top-3 match + eval range ±100cp)
- **Scope:** apply to labeled_1m.shard (1M → ~500K filtered)
- **Purpose:** data-quality test (filtered vs unfiltered at same size)

### C4: BC rerun on final binary
- **Purpose:** same-conditions comparison for the H1 table
- **Status:** queued

---

## Phase D: DiffuSearch Implementation (NEXT MAJOR ITEM)

**The single highest-impact change:** replace the single-forward-pass
policy with a discrete diffusion policy that predicts future game
states through iterative denoising.

### D1: Data pipeline extension
- Extend the shard worker to emit **horizon-4 trajectories** (current
  state + 4 future state-action pairs) from each game position
- New shard format v2: variable-length records with a trajectory block
- Each trajectory: [state(77 tok), action(1 tok)] × 4 steps

### D2: Model architecture change
- Remove causal attention mask (bidirectional attention)
- Extend max sequence length to ~420 tokens (77×5 source states + 8 moves)
- Add a [MASK] token to the vocabulary
- Add a source mask (state tokens frozen, future tokens diffused)

### D3: Training loss
- Absorbing discrete diffusion: randomly mask target tokens with
  probability (t+1)/T at timestep t
- Loss: cross-entropy on masked tokens, weighted by λ_t = 1 - t/T
- The model learns to denoise: given a noisy future, predict the clean
  future, conditioned on the current state

### D4: Inference
- Progressive denoising: start with all future tokens as [MASK]
- For t = T-1 to 0: forward pass → predict x₀ for each masked token
- Select 1/t fraction of the least confident tokens to keep masked
- The rest are fixed to the model's prediction
- After T steps: extract the first action from the denoised sequence

### D5: Gate
- Token accuracy: first-move prediction accuracy ≥ 40%
- Puzzle accuracy: WAC solve rate ≥ 30%
- SPRT: Elo vs BC-v0 → expect +100–500 based on DiffuSearch paper

### Why this should work
The DiffuSearch paper (arXiv:2502.19805) demonstrates:
- +540 Elo over a one-step policy with the SAME architecture (7M params)
- +14% action accuracy over MCTS-enhanced policies
- +30% puzzle-solving improvement
- The gain comes from **bidirectional attention over future tokens**
  which enables implicit search through the self-attention mechanism

### Risk: our implementation differs from the paper
- Our token representation is piece-codes, not FEN text
- Our data is BC (played moves), not SF-optimal trajectories
- These differences might reduce the gain — but the DiffuSearch paper
  used the same GPT-2 backbone, suggesting the technique is robust to
  representation choices

---

## Phase E: M2 Engineering (parallel with D)

| Item | Scope | Priority |
|---|---|---|
| INT8 parity fix | Accumulator-level bisect to find the offset bug | Medium |
| Pin-aware movegen | Replace copy-make verification; perft green | Medium |
| Perft init-touch | Touch BSS tables at startup; perft wall ≈ CPU time | Low |
| Latency table | p50/p99 for reference FP32 and INT8 paths on i5-9400F | Medium |

## Phase F: M3 Ablations (after D)

| ID | Question | Method |
|---|---|---|
| H2 | Parameter knee? | Sweep 3M/8M/20M/50M at fixed budget |
| H4 | Trunk: attention vs SS2D vs conv? | 3 trunks, fixed budget |
| H5 | Depth: tied-block k-curve? | k=1,2,3,4 unrolled |
| H6 | Filter value? | Legality / hang / pin / rep / 75-move ablation |
| H8 | Input planes (attack/defense/x-ray)? | ± features, fixed model |

## Phase G: M4 Engine B (after M3)

- Micro-eval QS with SEE-lite
- Node-budget sweep (200/2K/20K/200K)
- Polyglot book (separate Zobrist constants)
- Syzygy at root (already have Fathom via CPM)
- Contempt λ, clock heuristic, tactical-density gate

## Phase H: v0.2/v0.3 Release

- v0.2: BC-v1 SPRT results + research map update
- v0.3: DiffuSearch implementation + SPRT + research map update
- Each release: tag, release notes, README results table update

---

## Honest Assessment: Are We Fully Implementing What We Planned?

### What we planned and DELIVERED ✅

| Item | Proof |
|---|---|
| Board core, perft to d6 | CPW suite green, both slider paths |
| Syzygy root probe | Conversion suite passing |
| Time management | No forfeits in 200-game match |
| SPRT harness + pool | 400+ games played across matches |
| 50M shard pipeline | Jul+Aug combined, 50M records |
| Lc0 self-play shard | 3M positions from 54K games |
| Reference FP32 inference | Parity 5e-6 vs PyTorch |
| INT8 NetQ (architecture) | Built, runs, wired to engine |
| H1 negative result | Documented with SPRT provenance |
| BC baseline (38–40% vs SF16-d1) | SPRT provenance |
| Clean-room protocol | Provenance log, no GPL source contact |
| v0.1 release | Tagged + pushed + CI green |

### What we planned but did NOT deliver yet ⏳

| Item | Why | When |
|---|---|---|
| BC-v1 SPRT results | Training was restarted after crash | ~16h from restart |
| DiffuSearch implementation | Deep-read done, spec written, implementation queued | Phase D |
| INT8 parity | o≥1 bisect continues | Phase E |
| Pin-aware movegen | M2 item | Phase E |
| Latency table | Needs INT8 first | Phase E |
| M3 ablations | After M2 | Phase F |
| 5M labeling | Killed per early-stop; relaunch if H1 positive | Phase D branch |

### What we planned and DELIBERATELY REJECTED ❌

| Item | Why rejected |
|---|---|
| SF depth-16 label distillation | H1 proved it degrades play by ~309 Elo |
| SF depth-16 as primary technique | Clean-room constraint + H1 evidence |
| DEQ / Anderson acceleration | No gain at k≤3 scale; parked behind H5 trigger |
| MPNN / graph networks | No evidence of Elo gain over transformer |
| SDF king safety | Learned features dominate hand-crafted fields |
| BVH quadrant pruning | One AND with slider mask is cheaper |
| CAT way-locking | Server-only; tournament HW won't expose it |
| Taylor Δ-state caching | Side-to-mover flip inverts encoding; exact accumulators better |

### Deliberately PARKED (with trigger conditions)

| Item | Trigger to unpark |
|---|---|
| Ternary quantization | INT8 parity lands AND ≤30 Elo loss measured |
| DPO vs student engines | After M2 baseline exists |
| DEQ with Anderson | H5 shows gains persist through k≥6 |
| NNUE-style accumulators | If we adopt an NNUE-flavored first layer |

---

## The Honest Ceiling

| Configuration | Estimated Elo (SPRT) |
|---|---|
| BC-v0 (current best, 20M data) | ~2000 (38% vs SF16-d1) |
| BC-v1 (50M data, current training) | ~2050–2100 |
| + DiffuSearch (implicit search) | ~2100–2300 |
| + Lc0 fine-tune (quality data) | ~2200–2400 |
| + Combined best practices | ~2300–2500 |
| DeepMind reference (270M, 15B annotations) | ~2895 |

The 2060 is the binding constraint. Scaling to DeepMind's 270M
parameters requires ~$300–600 of cloud GPU time — worth considering
after the local pipeline is proven.

---

## Open Questions (no action, documented for the record)

1. Does the H1 negative result hold at larger data volumes?
   → Test: label 5M at depth 16, train, SPRT. Expensive but definitive.
2. Does the o≥1 NetQ divergence have a single root cause?
   → Likely: a blob parse offset. The o=0 exact match narrows it to
     a per-row issue. Fresh eyes needed.
3. Does the mask sidecar scale to 50M records?
   → Test: generate + measure. Expected ~10 min (I/O bound).
4. Does the Lc0 self-play data have different properties than BC data
   beyond the target difference?
   → H8 test: compare loss curves on held-out BC vs Lc0 data.
