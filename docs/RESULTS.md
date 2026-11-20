# RESULTS — Measured Verdicts (living ledger)

All verdicts: fastchess SPRT vs SF16.1 (60+6, openings.epd, both colors),
unless noted. Elo ± CI from the game-level MLE; pair-model numbers where run.

## Chain 0 — baselines (v0.1 era)

| Net | Data | vs SF16-p1 | vs SF16-p2 | Notes |
|---|---|---|---|---|
| BC-v0 | 20M human, 2 ep | **−83 ± 33** | −72 ± 37 | 200 games each, rerun-replicated (−83/−72) |

## H1 — SF-label distillation (2026-09, v0.1 era)

| Net | Data | vs SF16-p1 | Verdict |
|---|---|---|---|
| Distilled-1M | 1M SF-d16 labels, uniform loss | **−436** | REJECT: uniform hard-label imitation poisons searchless play |

## Chain 1 — data scaling + mechanism fixes (2026-10-04/05)

| Experiment | Change | vs SF16 | Verdict |
|---|---|---|---|
| **BC-v1** | 50M human (2.5×), 3 ep | p1: **−32 ± 35** (first run, full 200); p2: **−7 ± 34 (49.0%)** | **DATA SCALING CONFIRMED**: +51 Elo over BC-v0 with non-overlapping CIs; near-parity with a 2-ply-searching SF16 |
| **H2** | identical labels, decisiveness-weighted loss | p1: **−280 ± 121** (24-game early stop) | **MECHANISM CONFIRMED**: +156 vs the −436 uniform baseline. SF labels aren't poison; uniform loss on them was. Still below BC — imitation precision gap persists (AV phase attacks it properly) |

## GRPO throughput — measured cost, and the real blocker (supersedes prior estimate)

Three separate things were conflated. Measured, in order:

**1. A real hang (fixed, committed).** `SFEval` blocked on `stdout.readline()`
with no liveness check; a forked grandchild inheriting the pipe means a dead
engine never EOFs. The first production launch sat 8 h at ~15% CPU without
reaching step 5. Every read now checks `poll()` first, and the caller respawns
the engine and scores the move 0. Log cadence is every step (was every 5), with
elapsed minutes, because a 5-step interval on a slow box is indistinguishable
from a hang.

**2. Actual SF cost (measured, one engine, trivial opening positions).**

| depth | ms/analysis | rel. to d12 |
|---|---|---|
| 6 | 4.6 | 17x |
| 8 | 6.9 | 11x |
| 10 | 14.1 | 5.6x |
| 12 | 78.4 | 1x |

Production config is `groups=128 x k=16` = **2048 analyses/step**. At depth 10
that is ~29 s/step, so **1500 steps is ~12 h of engine time**, not "days to
weeks". The earlier 30x cost estimate was wrong and is retracted.

**3. The actual blocker: this box is shared and memory-exhausted.** Measured
while a reduced probe ran: `load average 41`, **0.0% CPU idle**, 363 MiB free
RAM with **31.5 of 32 GB swap in use**, and the three SF children holding only
21 s of CPU across 4 min of wall time. The load is five concurrent agent
sessions, not background noise. On a free box step 1 completed in **18 s**; the
very next step stalled because the box filled.

**Conclusion:** GRPO is *not* intrinsically too slow. It is unrunnable *here*,
right now, because the machine is oversubscribed and swapping. Correct prior
note: an earlier revision of this entry blamed D-state I/O load and quoted
steady 14% CPU per engine. Both were wrong -- `ps %cpu` is a lifetime average,
not instantaneous, and the real cause is CPU saturation plus swap thrash.

**Unblock conditions:** an idle box (or `sf_pool` sized to genuinely free
cores) and depth 10 rather than 12. No GRPO verdict exists.

## Verdict B FINAL (2026-10-11): REFUTED at 800 games — stage 2 ≈ stage 1

The 800-game confirmation (B-confirm chain, fastchess 15+0.2, openings epd,
both colors) overturns the 300-game reading:

- raw: 166W / 151L / 483D = 50.94% for stage 2
- engine-name pair-model recount: **pair score 0.4906, Elo −6.5 ± 12.3**

The 300-game +18.55 ± 24.16 (~0.8σ) was sampling noise, exactly what the
confirmation was armed to test. **Stage 2 (d16 RCT/QAT) provides no
measurable gain over stage 1** — the RCT/QAT recipe is struck from the
promotion path. GRPO's base choice (s2) is unaffected: its SPRT compares
against its own base, and s2 ≈ s1 anyway.

The 300-game lesson stands as process: sub-1σ signals get confirmation runs
before they steer decisions.

---
## Verdict B confirmed and sharpened (2026-10-09): stage 2 +18.55 ± 24.16 (300 games) — SUPERSEDED by the 800-game refutation above


The +18.55 figure is CONFIRMED by a direct engine-name recount of the PGN
(av-s2-vs-s1.pgn): stage2 scored 158/300 = 52.67% -> +18.55 Elo, result
tokens all clean (68 × 1-0, 62 × 0-1, 170 draws). Still only ~0.77sigma --
an 800-game confirmation is armed and queued behind the GRPO phase.

Two traps documented so this does not resurface:

- **fastchess txt perspective.** The final txt block for that match reads
  `Elo: -18.55` -- from stage1's perspective, because av_phase listed stage1
  first on the command line, while the PGN's first game has stage2 as White.
  The txt Elo perspective follows CLI order, NOT PGN White order. Recount by
  engine name from the PGN; treat txt perspective as untrusted.
- **sprt_pentanomial.py had an engine-identity bug (fixed, d63dd50).** It
  scored every game from White's side and never checked which engine held
  which color, reporting White's aggregate (51%) as engine1's: +6.9 ± 20.1
  where the true pair-model answer is +18.5 ± 20.1. Fixed and validated
  against the direct recount: pair score 0.5267 = 158/300 exactly.

## GRPO "stall" root-caused: per-op latency inflation from box contention (2026-10-09)

Seven hours of the production run produced exactly one step. Diagnosis chain,
for the record -- py-spy is unusable here (ptrace_scope=1), so the tools were
`strace -f` on a probe plus `faulthandler.register(SIGUSR1)` dumps:

1. The probe reproduced the pattern deterministically (step 1 fine, step 2
   crawls). strace showed the main thread completing `bestmove` reads and then
   going silent -- no syscalls for minutes while in state R: userspace spin
   inside CUDA ops.
2. Repeated SIGUSR1 dumps showed the frame ADVANCING across the whole step
   (sampler line 218 -> update forward 295 -> rewards 274 -> forwards in
   model.py). Nothing is stuck; every single operation is slowed.
3. Microbenchmarks in the same windows are fast in isolation (0.1 ms per tiny
   CUDA op; 12 ms per depth-10 SF eval), but a tiny 6-step probe needed
   35+ min for ONE step while a 6-step groups=2/k=4 probe finished in 0.2
   minutes at load 45. Cost scales with ops per step: production is
   2048 SF evals + ~500 CUDA launches per step, each inflated to
   seconds by sibling sessions (CPU load 45 with swap storms, GPU shared
   with three other CUDA contexts). One step then takes hours.

**The trainer code is exonerated.** The liveness/respawn path even proved
itself during diagnosis: when two engines died, both were detected and
respawned correctly.

Countermeasures shipped (data repo c12bce8, e912314):
- grpo_train.py resumes from the newest grpo_s*.pt in --out.
- grpo_phase.sh requires a GPU with zero foreign compute processes, runs a
  90-min step-rate watchdog (fresh per-run log; kill on 90 min without a new
  step line; export-every 100 bounds checkpoint loss), and a re-armed phase
  continues from checkpoints instead of restarting.

The honest consequence: production GRPO completes only in genuinely quiet
windows. The gates refuse degraded ones, the watchdog escapes mid-run
degradation, and resume means progress accumulates across windows.

## Tooling state (2026-10-10)

- **d16 TB sidecar: complete.** Full-shard regen finished (500k scanned,
  3,728 TB labels vs the previous partial 2,653/335k-prefix), atomic-rename
  written, sample-validated (schema, ≤5 pieces, sorted idx 897..499428).
  Ready for the next AV stage-2 cycle.
- **GRPO + B-confirm: still gated.** The shared box sits at load ~13–14
  (sibling agent sessions at 60–77% CPU each); the phase gates require
  load ≤ 6, thrash ≤ 50 KB/s, zero foreign GPU processes. Correctly waiting —
  launching at load 14 means ~3x-slowed steps (measured earlier: the box
  inflates per-op latency 30–50x at load 40+).

## GRPO v1 FINAL (2026-10-12): CATASTROPHIC REGRESSION — reward hacking confirmed

The 1500-step run completed (15 checkpoints) and the full 400-game SPRTs
give an unambiguous verdict:

- **GRPO vs base: 31/400 = 7.75%** (base scores 92.25%)
- GRPO vs SF16.1-p1 pool: 13.5/400 = 3.38%
- Games ending in 2–6 plies; zero-draw stretches — degenerate play

**This is textbook reward hacking.** The training reward (single depth-10
SF eval of the moved-to position, r_mean) climbed 0.25 → 0.57 while real
play collapsed to 7.75%. The policy learned to steer the evaluator, not to
play chess. Two design failures compounded:

1. **KL control was absent in practice.** kl_beta 0.03 let divergence grow
   exponentially: 2.6 nats @ step 350 → **371 nats @ step 1500**. Healthy
   RLHF keeps 5–15 nats with adaptive penalties. By step 350 the run should
   have been stopped or the coefficient raised.
2. **The reward is hackable by design.** A single absolute SF eval rewards
   evaluator-steering; real chess RL uses game outcomes or eval-deltas.

The intermediate SPRT "Elo: inf, 22-0" lines in the log were intermediate
SPRT evaluations at tiny game counts — the full 400-game engine-name counts
above are the verdict.

**Salvage sweep COMPLETE (2026-10-12): no checkpoint survives.**

| ckpt | KL | score vs base (100 games) |
|---|---|---|
| s100 | 0.55 | 48.5% |
| s200 | 1.6 | 46.5% |
| s300 | 2.6 | 38.0% |
| s400 | 4.4 | 44.0% |
| s500 | 5.8 | 45.5% |
| s1500 | 371 | 7.75% |

Every checkpoint scores below 50% — the RL was neutral-to-harmful from the
FIRST checkpoint, long before the KL collapse. The static-eval reward was
gameable from step 1; KL control could not have saved it.

**GRPO v1 final verdict: RL against a static SF evaluator is refuted.**
Strength-oriented RL on chess requires game-outcome rewards (self-play
win/loss at scale) — a fundamentally different and far more expensive
architecture. The failure-mode analysis (reward hacking dynamics, the
nan_to_num/exp backward leak, the KL trajectory) is the durable deliverable.

## Self-play outcome learning — loop closed (2026-10-13)

Two iterations of the full loop (self-play generation -> PGN->shard ->
outcome BC -> SPRT), both NULL at game level:

- Iteration 1 (s4): 10k games, 1 epoch on 53k winner-move records: +2.32
  ± 3.19
- Iteration 2 (s5): 30k games total, 4 epochs on 155k records, init from
  s4: −1.16 ± 3.93; draw rate 74% (healthy chess, unlike GRPO's degenerate
  games)

**Why it is information-thin:** in self-play between identical engines,
winner-side and loser-side moves are distributionally identical up to
opening luck and in-game variance. The outcome labels which games were
won, but the moves themselves carry no new chess knowledge to imitate --
the policy just re-learns its own move distribution.

**What would change the picture (all far above this box's envelope):**
true policy-gradient RL (REINFORCE/PPO) on game outcomes with proper
advantage baselines at 100k+ game scale; or search-augmented training
(MCTS-guided policy improvement). Both are architecture-scale projects.

**Positive deliverable:** the complete outcome-learning loop (self-play
37 min/10k games with -recover stall handling, roundtrip-validated
PGN->shard converter, NaN-guarded trainer, SPRT harness with labeled
perspectives) runs unattended in ~1 h per cycle. The infrastructure is the
asset; the pilot verdicts are its calibration.

## Pending (armed chains)

| Experiment | Gate |
|---|---|
| bc_v1f quality-FT SPRTs | running (training epoch 2) |
| keep-best e0/e1/e2 ranking | after bc_v1f |
| LoopCD grid (R×α) | after keep-best |
| INT8-vs-FP32 h2h | after LoopCD |
| DiffuSearch a0-match | gate 0.25 / double-down 0.40 |
| AMZ puzzle gate | base + 1.5 pts |
| AV two-stage vs BC-chain best | after labels (~2 days) |
| GRPO 1500 steps (armed, waiting for idle box) | d16 shard, depth 10, thrash/load-gated; auto-SPRTs vs base then pool; promote only if beat base AND no pool regression. Corrupt NaN-era checkpoints/SPRTs deleted 2026-10-09 |
| d16 TB sidecar full regen | **DONE + VALIDATED** (2026-10-09/10): 500,000 records scanned, 3,746 probed, 3,728 labeled (was 2,653 / 335k prefix), atomic rename; independent check: 200 random labels re-decoded from the shard and re-probed — idx alignment, WDL, move legality all match (200/200) |
| d10 TB sidecar | **FULL** (verified 2026-10-10): 15,186 labels, idx 897..1,999,623 of ~2M records — complete, no regen needed |
| B-confirm: s2 vs s1, 800 games | **DONE — VERDICT B REFUTED** (2026-10-11): 166W/151L/483D = 50.94% raw; engine-name pair-model recount **−6.5 ± 12.3** — the 300-game +18.55 ± 24.16 was noise. s2 ≈ s1; no promotion. GRPO-vs-base SPRT remains internally valid regardless of base choice |
| Self-play loop iteration 2 (s5) | **DONE — NULL** (2026-10-13): s5 (4 epochs on 155k records from 30k s1+s4 self-play games, init from s4) vs s1 = 49.83% (−1.16 ± 3.93 game-level, +1.2 ± 20.1 pair-model), 74% draws (healthy games — no GRPO-style degeneracy). Two iterations of winner-move self-imitation: both null. The outcome signal in self-play between identical engines is information-thin: winner-moves and loser-moves are distributionally identical up to opening luck |
| Self-play outcome BC pilot (s4) | **DONE — NULL, SAFE** (2026-10-12): 10k self-play games (2,696 decisive), winner-move BC 1 epoch; s4 vs s1 = +2.32 ± 3.19 game-level, −2.3 ± 20.1 pair-model. No gain, no collapse — in contrast to GRPO's 7.75% catastrophe on the same base. Infra proven: 10k games in 37 min, PGN→shard converter roundtrip-validated, full loop < 1 h. Scale-up path: iterate self-play with the improved net + outcome-weighted loss + more epochs |
| AV s3: full-sidecar A/B | **DONE — NULL** (2026-10-11): s3 = 144.5/300 (48.17%, −12.75 ± ~20) vs s2 by engine-name count; pair-model +12.7 for stage2 (White of game 1) agrees. The fuller TB sidecar (3,728 vs 2,653 labels = 0.75% of the set) changes nothing measurable. **Positive: first end-to-end trainer run to complete and produce a playable net** (NaN-guarded trainer, c4c8cda lineage) |

## Mechanism baselines (E4) — NOT cross-comparable (different opponents)

| Net | Opponent | Conversion (≥3 mat, move 10+) | Replays/game | Forfeits |
|---|---|---|---|---|
| Distilled-1M-ext | SF16-p1 | 77.9% | 0 | 0 |
| **BC-v1** | SF16-p2 | **45.3%** (72/159) | 0 | 0 |

BC-v1's low conversion vs distilled is confounded by the stronger defender
(SF-2ply defends better). E1 Syzygy labels target exactly this bucket; the
AV phase re-measures on a fixed opponent.

## Pair-model (pentanomial) cross-checks

| Match | Pair Elo | Consistency |
|---|---|---|
| BC-v1 vs SF16-p2 | +3.5 ± 24.6 (49.5% pairs) | ✓ confirms game-level −7 ± 34 |
| Distilled-1M-ext vs SF16-p1 | +6.9 ± 24.6 | ✓ |

## Scope change — labeled set halved for time-to-verdict (2026-10-06)

The mixed-depth labeling target was cut from **4M@d10 + 1M@d16 to 2M + 0.5M**.

Reason: the box is shared and other sessions drove load average to ~47 on 6
cores, holding the labeler at 23 pos/s versus its 62 pos/s baseline (d16
historically ran at 8-9 pos/s). That projects the original ~49 h to ~5 days
before AV could start. Halving brings it to ~2.5 days at current load, and
much less if the box frees up.

Why this is cheap: it is a **fine-tune**, not training from scratch, and it
saturates well before 4M — 2M x 2 epochs at batch 512 is 7.8k steps, and the
tablebase sidecar still sees ~60-160k exact endgame labels from the <=5-piece
positions in a 2M sample. Label *depth* is preserved, and depth matters more
than volume for a fine-tune. If the AV verdict later shows the model is
data-limited rather than mechanism-limited, the labeler resumes to 4M (the
shard is durable and `--resume` continues from the record count).

**This decision was only affordable because labeling became crash-safe**
earlier the same session: the relaunch logged
`resuming at record 130000 (130000 already durable)` instead of restarting.

## Loss attribution — where the Elo actually goes (2026-10-06)

`trainer/analyze_losses.py` walks each net move, evaluates before/after with
SF (side-to-move POV, reads to `bestmove`), and attributes each loss to the
net's own worst moment. Mate positions are excluded: a mate delivery is the
end of a game, not its cause. 40 games per match, depth 10.

| Match | W-L | tactics | endgame | material | other | phase | median drop |
|---|---|---|---|---|---|---|---|
| BC-v1 e2 vs SF16-p1 | 12-14 | **64%** | 0% | 9% | 27% | middlegame 6, opening 4, endgame 1 | 325cp |
| BC-v1 e2 vs SF16-p2 | 6-16 | **40%** | 30% | 10% | 20% | middlegame 5, endgame 3, opening 2 | 318cp |

**Reading:** losses are dominated by *tactical* collapses in the middlegame
(a ~320cp median drop is a hung piece or a missed threat). Endgame errors are
real but secondary — and the endgame is where tablebases already give exact
play. Openings barely register.

**Implication — this validates the armed plan rather than redirecting it:**
- decisive-position weighting already targets this and measured **+156 Elo**
- GRPO (armed) optimizes exactly this: sharpness against a strong opponent
- do **not** spend the next cycle on opening books, endgame engineering, or
  material heuristics — together they are ~15-20% of the attributable losses

It also says the searchless ceiling is a *tactics* ceiling, so any future
mechanism should be judged on middlegame tactical accuracy, not endgame or
opening metrics.

## MirrorAvg — inference-time test-time augmentation (2026-10-06, INCONCLUSIVE)

Mechanism: the policy head is indexed by absolute squares and the square
embeddings are not mirror-equivariant, so the net's file-mirrored view
disagrees systematically. `MirrorAvg` sums a move's score with the score of
its mirror image (u,v -> u^7,v^7) and averages the WDL head. Still O(1) in
search depth; one extra forward pass (~10 ms against a 15 s clock). UCI
option, **default off**, so no running verdict is affected.

Measured with a paired position-level test (`tests/mirror_ab.py`, 250 real
positions from the BC-v1 p2 match, both engines on one process, SF depth 8):

| metric | value |
|---|---|
| positions where the move changes | 17/250 (**6.8%**) |
| paired eval difference (on - off) | **+38.1cp** |
| 95% bootstrap CI | [-46.9, +142.9] |
| better / worse | 9 / 7 |

### Re-measured at n=2500: **HELPS**

| measurement | n | result |
|---|---|---|
| paired eval on changed moves (BC-v1 e2) | 180 changed positions | **+23.4cp, 95% CI [+5.3, +44.6]**, 89 better / 83 worse |
| policy accuracy vs SF-best, v3 AV net | 800 positions | 0.422 → **0.435** with MirrorAvg (+1.3pp, ~0.75 s.e.) |

The move changes in 7.2% of positions; the paired interval excludes zero, and
an independent metric on a v3 net moves the same direction. The effect is small
(~+1.7cp averaged over all positions) and free: one extra forward pass, ~10 ms
against a 15 s clock, and still one decision per move.

### CORRECTED (2026-10-08): the game verdict was read backwards — MirrorAvg HURTS

A PGN recount of the same 400 games (counting points by engine NAME rather
than trusting the reported Elo line) shows the opposite of what was recorded
below: **plain 220.0 (55.0%) vs mirror 180.0 (45.0%)**. fastchess reports Elo
from the FIRST engine's perspective; `+34.86` meant *plain* (MirrorAvg off)
was better. MirrorAvg therefore **hurts by roughly 35 Elo**, and the default
was reverted to OFF.

The uncomfortable part: two independent cheap measurements — the paired
eval delta (+23.4cp) and the v3 policy-accuracy delta (+1.3pp) — both pointed
the WRONG way while agreeing with each other. Cheap proxies are not evidence
of game outcomes, however much they correlate on the sample that generated
them. A cross-check of the sign convention (the phase_post net-vs-SF matches
reporting −32 for the net, matching its established rating) would have caught
this before the default flip; that check is now a standing rule for every
verdict: **recount the PGN by engine name before believing a direction.**

### Original (misread) entry kept for the record

The full 400-game match (bc-best vs bc-best with MirrorAvg, same weights,
same openings): 85W/45L/270D, pair-score 55.0%. The paired position-level
prediction (+23.4cp CI [+5.3,+44.6]) pointed the right way and the game
verdict is now 1.8 s.e. from zero — a real, cheap gain. MirrorAvg should be
**enabled by default** for all subsequent nets; the only reason it has not
been flipped in the running binaries is that the keep-best gate was mid-flight
when the evidence landed, and verdict A had to stay clean.

**Decision: enabled — but deliberately not yet.** Flipping the default while
`keep_best`'s gate is mid-flight would give some matches MirrorAvg and others
not, and flipping it before verdict A's SPRT would confound that comparison
with the AV recipe. Sequence instead: verdict A runs clean (both sides without
MirrorAvg), then a dedicated SPRT of `bc-best` vs `bc-best + MirrorAvg` gives
the Elo number. Everything needed is in place: mapping-verified
(`tests/mirror_consistency.py`), CI-gated, and now measured twice.

The earlier n=250 run read +38cp with a CI spanning zero; the point estimate
fell toward +23cp as n grew, which is what one expects when an initial small
sample was noisy — the direction held.

## VERDICT A — CORRECTED: the AV bundle did NOT beat the BC chain (2026-10-07/08)

The 400-game match numbers were first reported with the sign backwards
(fastchess Elo is from the FIRST engine's perspective; bc-best was listed
first). Corrected reading:

| measure | value | correct interpretation |
|---|---|---|
| game-level (400 games, bc-best 76W/61L/263D) | **+13.03 ± 17.89 for bc-best** | the BC chain is ~+13 over the AV bundle (within noise) |
| pentanomial pair model | **+4.3 ± 17.4 for bc-best** | same direction, same noise |

So verdict A is a **null-to-slightly-negative**: the AV bundle (v3 bucketed
value head, TB labels, decisive weighting, opening upsampling, Muon/WSD
fine-tune on 2M d10 labels) did not measurably improve on the BC chain it
warm-started from. Given 263 draws the CI is tight; this null is well-measured.

**Verdict B, correctly read**: the d16 RCT/QAT refinement (av-s2 vs av-s1,
300 games, stage-1 listed first) reported **−18.55 ± 24.16** for stage 1 —
i.e. the refinement is **+18.6 ± 24.2 over stage 1**. Positive direction, not
significant at 300 games, but it is the ONLY mechanism measurement of the
session that points up at game level. RCT/QAT on deeper labels is the
strongest candidate for a properly powered confirmation (500-800 games).

## VERDICT A — original misread entry (kept for the record): "the AV bundle works: +4 to +13" (2026-10-07)

First mechanism-level verdict of the project. The AV stage-1 bundle — v3
architecture (material-bucketed value head + king-quadrant embedding),
syzygy-rescored TB labels, decisive-position weighting, opening-book
upsampling, mirror augmentation, EMA, Muon + WSD — fine-tuned on 2M d10
labeled positions from the BC gate winner, measured over a full 400-game match:

| measure | value |
|---|---|
| game-level (400 games, 76W/61L/263D) | **+13.03 ± 17.89** |
| pentanomial pair model | **+4.3 ± 17.4** (pair score 50.6%) |
| conversion (winning material) | 0.301 |

Positive on both models, and the early-stop read (+34.86 at 20 games) points
the same way. This is the first net to beat the BC chain rather than lose to
it — every prior full-mechanism attempt (distillation, DiffuSearch, AMZ)
measured negative.

Worth noting what the bundle did NOT contain: castle/EP/rating embeddings were
inert during training (see below) and HiCo history was never applied, so the
measured gain comes from exact TB endgame labels, decisive weighting, opening
upsampling, and the bucketed value head. The state inputs are untested upside.

Phase B (the d16 refinement: + RCT recycling + QAT projection on 500k d16
labels) is waiting on the remaining d16 labels and will measure against this
stage-1 net — verdict B isolates whether the refinement adds anything.

## BC-v2 — data scaling is exhausted in this regime (2026-10-07)

Built the largest corpus the box could hold: the lichess archives were
originally capped at 25M positions/month, so the same 56 GB yielded 80M more
records (40M/month) for a **130M-position** corpus, then trained bc_v1's exact
recipe on it — with one deliberate change: **1 epoch over 130M unique** shows
130M positions against bc_v1's 150M over 3 epochs, so the comparison is
compute-matched and isolates data diversity.

| run | corpus | positions seen | verdict vs BC-v1 gate winner |
|---|---|---|---|
| BC-v2 | 130M unique | 130M | game-level **+2.61 ± 18.11**, pair model **−6.1 ± 17.4** (400 games) |

**Statistically a dead heat.** Two conclusions:

1. **The +51-per-10x scaling trend does not continue here.** 2.6x more unique
   data bought nothing measurable. bc_v1's recipe was already at the
   data-quality ceiling for this corpus (Lichess games, SF-d10/d16 targets) —
   more data of the same kind is not a lever anymore. Future Elo must come
   from mechanisms or from *better* labels (deeper/more accurate), not more of
   the same.
2. **No regression either**: bc_v2 saw each position once and matches a net
   that saw its data three times. The 70% draw rate (281/400) keeps the CI
   tight, so this null is well-measured, not underpowered.

Practically: the 7.75 GB corpus and mask sidecar are kept (they are the
natural substrate for the next mechanism-level run), and the GPU was handed
back to the AV verdict, which the same watcher had queued behind it.

## DiffuSearch — REJECTED by a strength test its own gate could not see (2026-10-06)

The moonshot's verdict metric was **a0-match**: does the denoised policy
predict the move a human played? Measured three ways on the epoch-0
checkpoint (loss 0.0076 on the training objective):

| measurement | value | what it actually measures |
|---|---|---|
| training-time `a0 match` printout | **0.971** | **invalid.** It masks only half the target region, so the future state tokens stay visible and the move is trivially recoverable from the visible resulting position |
| honest a0-match, whole target masked, T=16, legal gate | **0.330** | next-move predictability from the position alone (0.290 without the gate) |
| **strength** — diffusion policy vs greedy play of the *same* BC weights | **0W 0D 12L** | playing strength |
| harness control — greedy BC vs greedy BC (identical policies) | 3W 2D 5L, 0.400 ± 0.158 | confirms the harness does not favour either side |

**Independent corroboration** (`trainer/policy_accuracy.py`, 300 labeled
positions, both policies scored against the *same* reference — Stockfish's
best move, taken from the shard's target list, so no extra analysis):

| policy | agreement with SF's best move |
|---|---|
| BC greedy (the engine's own policy) | **51.0%** |
| DiffuSearch (a0, T=16) | **23.3%** |
| the two policies agree with each other | 35.3% |

The diffusion policy picks SF's best move less than half as often, which
explains the 0-12 result without invoking the playout harness at all. The two
measurements are independent and agree.

**The metric is validated against ground truth.** Scoring the existing nets
with it reproduces the ordering their game verdicts established, at about a
minute per net instead of hours of games:

| net | policy acc vs SF-best | known game verdict |
|---|---|---|
| bc_v1 e2 | 0.555 | −7 (best) |
| bc_v1 e0 | 0.530 | −32 |
| bc_v1f e2 | 0.500 | ~0 (noisy-neutral; overlaps e0 within noise) |
| bc_v0 e1 | 0.485 | −72 |

**Two methodology rules for the metric, both learned the hard way.** First,
comparisons must use the SAME position set: the first 300 shard records score
~0.51 for BC-v1 while the first 1000 score 0.461 — subset composition alone
moves the rate by ±5pp, which is bigger than most effects. On identical
positions (n=1000) the metric ranks verdict A correctly: AV stage-1 0.489 >
BC-v1 e2 0.461, matching +4-to-+13 Elo. Second, it is a *rejection* gate, not
a strength substitute: it caught DiffuSearch's 23% collapse, but a mechanism
expected to help only in a specific game phase (AV's TB endgame technique)
gains games without moving top-1 agreement much. Cheap filter first, game
verdict to confirm.

**This also calibrates the project.** BC-greedy sits at 51% against SF-best
and plays at roughly -7 Elo versus SF16; DiffuSearch sits at 23% and loses
every game. So *policy accuracy against a strong reference* is the metric
that tracks strength, it is cheap (no games required), and it is comparable
across policies. It replaces both a0-match against human moves and puzzle
scores as the gate metric going forward.

The strength harness (`trainer/diffusion_playout.py`) was validated before
its verdict was believed: its inference path reproduces `infer_diffusion`'s
a0 accuracy (27% vs 33%), and three separate harness bugs were found and
fixed on the way — a duplicate-SEP off-by-one that shifted the a0 slot
(agreement 0.08), a row 68 tokens shorter than the training shape (0.22), and
a denoising schedule that read a0 before revealing any context. The
0W-12D-0L result survived all of them.

**Verdict: REJECT.** A policy that agrees with the played move ~30% of the
time is far below the greedy policy of the same weights (which is roughly
SF16-parity at -7 Elo). Diffusion action accuracy does not convert into
playing strength here, and no amount of extra epochs on the same objective
addresses that.

**Consequences, which matter more than the verdict itself:**
1. The pre-registered gate (kill < 0.25, double-down >= 0.40) would have
   read 0.33 and spent ~9 more GPU-hours extending a rejected mechanism.
   `moonshot_queue.sh` now records the verdict and skips stages 1, 2 and 5;
   the AMZ pilot (an independent mechanism) proceeds as planned.
2. **A next-move-prediction metric is not a strength metric.** Every future
   mechanism needs a play-based or paired-eval verdict, not an accuracy
   against human moves. This is now the standing rule for new gates.
3. The training loop's eval was measuring a leaked quantity; it now prints the
   honest score and labels the old one as leaked.

Also fixed in the same pass (real bugs in the diffusion inference path):
`decode_source_board` dropped castling and en-passant, so `infer_diffusion`'s
legal-move gate **forbade castling outright** and mis-decoded EP states.

## AMZ — REJECTED: oracle supervision destroys the policy (2026-10-06)

The second moonshot, run to completion by the queue (50k positions, 1 epoch,
lr 1e-4), failed its own pre-registered puzzle gate and then the calibrated
metric:

| measure | BC base | AMZ pilot |
|---|---|---|
| puzzle score (queue gate: needs base + 0.015) | **0.460** | **0.162** |
| policy accuracy vs SF's best move | **51.0%** | **15.0%** |

One epoch of oracle supervision on 50k positions cut policy accuracy by
two thirds. The likely reason is structural rather than a tuning accident:
AMZ's targets are conditioned on the **opponent's reply distribution**, which
the policy never observes at inference. Training a memoryless policy on
reply-conditioned soft targets is privileged-information training, and the
mismatch shows up exactly where it would — the policy learns to expect
information it will not have.

That is a testable claim about the mechanism, not just a failed run: AMZ
would need the reply information at inference (i.e. search, which this project
excludes) or a way to marginalize it into a memoryless target. The first
moonshot's failure (action accuracy does not convert to strength) and this
one (privileged targets do not transfer) are both about the gap between a
training signal and what a searchless policy can actually condition on.

The 400-game SPRT that would have followed was killed rather than run: the
gate had already rejected the net decisively and policy accuracy measures the
same thing in seconds, so the games would only have taken CPU from the
labeler, which is the critical path.

**Both moonshots are now rejected.** The GPU plan reverts to the levers with
documented evidence behind them: data scaling (50M -> 130M, running) and RL
(AV, then GRPO).

## Three v3 state fields were inert: castle, EP, rating (2026-10-06)

Found while previewing the AV recipe on idle GPU. `train.py` calls the model as
`model(codes, side)` — it never passes castling, en-passant or rating, so those
embeddings are always fed index 0. Because they are zero-initialized, and
because embedding gradients only reach the row that was used, every other row
stays exactly zero **forever**. Verified on a real 2-epoch run:

| tensor | nonzero after training | interpretation |
|---|---|---|
| `castle_emb.weight` | 256 / 4096 | exactly one row (mask 0) |
| `ep_emb.weight` | 256 / 2304 | exactly one row (file 0 = none) |
| `rating_emb.weight` | 256 / 4096 | exactly one row (bucket 0) |
| `king_bucket_emb.weight` | 1024 / 4096 | four rows — **live** (computed inside forward from the board) |
| `hist_emb`, `hist_gate` | 0 | never applied: no history is passed |

So of the v3 additions only the material-bucketed value head and the
king-quadrant embedding ever contribute. There is no train/inference bug — the
engine passes the real values, and the corresponding weights are zero, so the
contribution is zero on both sides — but the fields cannot earn Elo because
nothing ever trains them.

`--state-aware` now feeds the record's real castle mask and EP file (EP index
= file+1, 0 = none) and is **off by default**, so the armed AV run keeps its
pre-registered recipe. With it, the trained rows go from 1 to 16 (castle) and
1 to 9 (EP), confirming the path is live.

**Measured effect at n=300** (1M-position preview, identical recipe, only the
flag differing): state-blind 0.507, state-aware 0.483, against a BC-v1 base of
0.510 — i.e. no gain, and the difference is under one standard error at this
sample size. A 2000-position rerun is in flight. The genuinely new information
is EP rights (the board cannot tell you a pawn just moved two squares);
castling rights are largely derivable from the board, so the ceiling here is
low. Queued as a candidate, not adopted.

## Volatility weighting — null at preview scale (2026-10-08)

Queued mechanism #1, now tested. `trainer/train.py --volatility-weight W`
upweights positions whose eval swung vs the previous ply (|delta| >= 150cp ->
weight W). Volatility is a free proxy for tactically-critical positions:
records are in game order, the delta needs no extra Stockfish work, and game
boundaries are rejected by a connectivity filter (sides alternate, board
differs in 2-4 squares, piece count within 1).

Measured with the preview A/B (1M positions, identical recipe, warm-start from
BC-v1 e2, same 1000 evaluation positions):

| arm | policy accuracy vs SF-best |
|---|---|
| AV bundle, no volatility | 455/1000 = 0.455 |
| + volatility weighting (W=2.0) | 459/1000 = 0.459 |

+0.4pp — inside noise. Honest caveats: a single design point (W=2.0/150cp), a
small preview corpus, and policy accuracy is known to under-credit tactical
robustness (verdict A's lesson). But the cheap filter is the reason it exists:
a null here does not justify game-level compute while verdict B and GRPO are
queued. The flag stays (default off, harmless), and the harness to re-test at
full scale exists.

Also fixed on the way: the flag silently never reached `build_batch` — the
prefetch refactor had moved the call site into the `_make_batch` wrapper and
the patch targeted the old inline call. The first A/B produced
**byte-identical checkpoints** (same md5), which is how the wiring bug
announced itself. When an A/B returns two identical artifacts, suspect the
plumbing before the mechanism.

## Verdict B — the d16 RCT/QAT refinement: +18.6 ± 24.2, the only upward game signal (2026-10-08)

300 games, av-stage2 (d16 fine-tune + RCT recycling + QAT projection) vs
av-stage1 (the d10 bundle). fastchess reported −18.55 ± 24.16 from stage 1's
perspective — the stage-2 net is ~+19 Elo up, within noise at 300 games
(1σ ≈ 24), but it is the only mechanism measurement of the entire session
that points UP at game level. Follow-up options: extend the match to 600-800
games for significance, or fold the RCT/QAT stage into the next full run and
let its verdict speak at scale. Phase B's TB sidecar was generated from the
335k-record prefix of the d16 shard (the labeler was still appending); the
remaining ~165k records carry no TB rows — harmless, but a full-shard sidecar
would be marginally stronger.

## GRPO — two NaN sources fixed; the first real launch was destroyed by them
(2026-10-08)

The first GRPO run at production size printed `pg nan kl nan` from the first
logged update and saved NaN weights (the later SPRT vs the SF pool — 0W/368L
— is simply what a NaN net scores; the promotion rule correctly refused it).
Two independent bugs:

1. KL(base||new) summed `blp.exp() * (blp - lp)` over the masked distribution:
   illegal squares are −inf in both, so the term is 0 × (−inf − −inf) = NaN.
   nan_to_num encodes the correct convention (p = 0 contributes nothing).
2. The k-padding fix left padded slots with −inf in both new and base
   log-probs, so their importance ratio was exp(−inf − −inf) = NaN, and
   NaN × adv(0) stayed NaN. Zeroed with nan_to_num.

Additional guards: non-finite losses are skipped (the update would NaN every
weight permanently), checkpoints are only saved when all weights are finite,
and the step log carries the skip counter. The earlier smoke test had exported
checkpoints DESPITE the NaN — "it exported" is not a finiteness check.

Also fixed the same day: the GRPO validation gate asserted legality on padded
dummy actions (index 0 = a1→a1 is never legal), killing the run at startup
whenever a position had fewer than k legal moves.

## Box-capacity reality (2026-10-08)

The shared box's external load (another session's Lean/Go/compiler builds
driving load average 20-47 on 6 cores) sets hard throughput limits:
GRPO at production size (groups=128, k=16, depth 12, 4-engine pool) ran at
5.1 s/step overnight and >25× slower under daytime load; the 130M-position
bc_v2 epoch took ~30 h at 36-40% GPU utilization because the python batch
pipeline starves under contention. Everything is sized and documented to
RESUME when the box frees; nothing needs re-decision.

## GRPO — compute-blocked, handoff state (2026-10-08)

All GRPO correctness bugs are fixed and verified finite at the real k=16
(validation gate with padding, KL nan_to_num, padded-ratio nan_to_num,
non-finite update skip, NaN-save refusal — see the entry above). The
production run (groups=128, k=16, depth 12, 1500 steps) was killed because
the box's external load stretches a step from 5 s (unloaded) to an estimated
17 min: 1500 steps would be 3-6 weeks. A reduced probe cannot answer the
learning question either — 120 steps at lr 1e-5 inside a 0.03 KL budget moves
the policy too little to read a reward trend.

**Resume procedure** (when the box frees up):
```
cd /home/wyatt/data && setsid nohup bash grpo_phase.sh >/dev/null 2>&1 &
```
It waits for nothing (av_phase is complete), warm-starts from the verdict-B
winner (`runs/av_v3_s2/net_e1.pt`), trains 1500 steps, and runs its own SPRTs
with the promotion rule (beat base AND no pool regression).

**Cost correction (2026-10-08, measured on the resumed run):** the "~2-3 h"
estimate in the first version of this entry was wrong by ~30x. Each step
needs 128 groups x 16 moves = 2048 REAL depth-12 analyses (4-engine pool):
~17 min/step under external load, ~9 min unloaded -> 1500 steps is roughly
**1-4 days depending on the box**, checkpointed at 500/1000/1500 so
intermediate nets can be evaluated at any time. A telling number from the
failed NaN run: it "finished" 1500 steps in 2 h because a NaN policy samples
garbage actions that skip the SF call entirely -- its speed was itself a
symptom. Expected first signal: whether r_mean climbs from its ~0.01
baseline. If r_mean stays flat for 500 steps, GRPO-as-configured is another
null and the SF-pool reward design (not the optimizer) is what to revisit.

## Queued mechanism candidates (2026-10-06)

Ranked by expected value per unit of compute, with the reason each is not
running yet. The loss-attribution result above (tactics dominate) is the
filter: anything that does not plausibly improve middlegame tactical
accuracy ranks low.

| # | Mechanism | Rationale | Why deferred |
|---|---|---|---|
| 0 | **DiffuSearch variants** (LID, CTAP, different horizons) | — | **dropped**: 23.3% policy accuracy vs SF-best (BC: 51.0%) and 0-12 in play; variants inherit the failure until that gap is explained |
| 0b | **AMZ variants** (coarser reply sampling, marginalizing the reply) | — | **dropped pending a mechanism fix**: reply-conditioned targets are privileged information a memoryless policy cannot use at inference |
| 1 | ~~Volatility weighting~~ | tested 2026-10-08: +0.4pp policy accuracy at preview scale (455 -> 459 / 1000) — inside noise | kept behind `--volatility-weight` (default off); re-test at full corpus only if tactical robustness proves to be the binding constraint |
| 2 | **MirrorAvg: confirm at game level** | measured twice now — paired +23.4cp (CI [+5.3,+44.6]) and +1.3pp policy accuracy on a v3 net; free at inference | needs a clean SPRT; deliberately sequenced AFTER verdict A so it does not confound the AV comparison |
| 3 | **Policy-aware hard-example mining** — upweight positions where the *current* policy's greedy move loses >= 200cp vs SF | the direct version of volatility weighting on the model's own errors rather than the game's | requires one inference pass per labeled position; `mine_blindspots.py` already covers the value-head variant and has never been run on real moonshot output |
| 4 | **Quantizer vectorization** (5x faster SPRT) | every future verdict gets 5x the games for the same wall clock | no mechanism value; pure infrastructure, revisit when a verdict is borderline |

Explicitly **not** queued: opening-book work, endgame engineering, and
material heuristics. Together they account for ~15-20% of attributable
losses, and the endgame already has exact tablebase play.

## Incident register (2026-10-04/05)

| Incident | Catch | Fix |
|---|---|---|
| parse_fen ep direction inverted | movegen fuzz (1/200) | direction flipped; perft green |
| phase_post `$txt` case typo | bash -x trace | fixed; 6 refs; memory guard added |
| Diffusion tokenizer: black pieces shifted ×2 + 'b' vocab collision | loss curve starting at 0.24 (impossible for honest CE) + device-side assert | CODE_TO_CHAR map + SIDE rename; poisoned 2M cache deleted; regression test added |
| Diffusion `ep >= 8` meant "no EP": every legal EP square (rank 3/6) collapsed to none | audit while restarting the moonshot; 0.13% of shard records carry a real EP square | token carries the file, rank implied by side-to-move; regression test pins all 16 legal squares |
| `codes_to_board` dropped castling + EP rights | same audit: neither appeared in `legal_moves`, so sample runs were silently truncated at every castling/EP move | takes both and sets them; castling + EP capture now generated |
| `_validate_sample_slice` read `s[2:2+64]` instead of `s[1:1+65]` | noticed while fixing the encoder: the gate that should have caught it inspected a shifted window (a1 dropped, side token included) | window corrected; EP token/side/rank invariant asserted in the gate |
| Sample store was ~13 GB (2M python lists) → OOM-killed | process died mid-scan with no output | chunked int16 (410 B/row, 820 MB); RSS logged per 200k records |
| OOM inside `torch.save` left a 0-byte cache; every later run died with `EOFError` | queue reported `FATAL: DiffuSearch training failed` immediately | `trainer/robust_io.py`: atomic writes (tmp+fsync+rename) + self-healing loads, used by all 10 trainer entry points |
| OOM killed `lo-data label` at 3.9M/4M records — 17 h of labeling lost (no output until completion) | shared box ran another session's Lean/Go builds to 25+ GB | batching + `--resume` (a partial output is a valid input prefix); RAM preflight in all chain scripts |
| `/tmp` cleanup deleted the python venv, killing the moonshot queue mid-run | `ModuleNotFoundError` from a running process | venv rebuilt at `~/.venvs/chess` (torch 2.6+cu124) with `/tmp/opencode/venv` as a symlink |
| `keep_best` ranked epochs by the FIRST `Elo:` line (a running SPRT snapshot) | same bug class as analyze_verdicts.py | `tail -1`; also fixed the `[ -z ] || [ ]` condition that could never pick a second epoch |
| AV phase looked for `net_best.bin.pt`, which nothing writes → silently fell back to `net_e2.pt`, ignoring the gate ranking | audit of the warm-start path | keep_best now publishes `net_best.pt`; av_phase consumes it |
| `/home/wyatt/data/*.sh` were unversioned | audit | committed under `latent-oracle-data/ops/` with the cascade topology and the live-script edit rule |
| **AV stage 1 crashed at startup**: `LambdaLR` cannot wrap the Muon `_JointOpt` wrapper, and the call was unguarded (the `try/except` meant to catch it sat *after* it) | dry run of the exact stage-1 flag set on a 20k labeled shard | `_ParamGroupSched` fallback so WSD actually applies; pinned against `LambdaLR` (which caught an off-by-one: it applies `fn(0)`, not `fn(1)`, at construction) |
| **AMZ pilot crashed** in its slice validation: samples are 4-tuples, unpacked as 3 | dry run of the exact queue invocation | unpack fixed, gate strengthened (board/side shape too) |
| **GRPO crashed** in the PPO update: positions with fewer than `k` legal moves gave ragged logp vectors (`stack expects equal size`) | dry run of the exact `grpo_phase` invocation | actions padded to `k` (dummy index 0 can never be legal), masked out of the loss |
| GRPO reward slots were filled with **NaN**, so any group with a padded slot had NaN mean/std → NaN loss | same dry run | zeros + explicit validity mask; mean/var computed over valid slots only |
| GRPO `build_step` appended to `fens` but not to `moves_per`/`acts`/`base_logp`, so the lists ran short and the reward loop indexed past their end | same dry run | every branch appends to all four lists |
| fastchess prints `Elo: X, nElo: Y` on **one line**, so `grep -oE 'Elo: ...'` also matches nElo — and `tail -1` selects exactly the wrong number. `analyze_verdicts` read a 6-game smoke match as −300.89 (nElo) instead of −190.85 | verdict-harness smoke run | `\bElo:` in Python; `sed 's/, nElo:.*//'` before extraction in `keep_best.sh` |

## 2026-10-17 — Clean label-depth test (d16 plain BC vs champion): BORDERLINE POSITIVE

ops/label_depth.sh, 300 games tc=15+0.2, av_v3_d16bc/net_e1_ema (fine-tuned
from av_v3 on the d16 shard, NO RCT/QAT/recycle/sidecar) vs av_v3/net_e1:
fastchess +36.03 ± 21.87 (d16bc perspective); pair-model recount
+36.0 ± 20.2 (1.78σ; pentanomial WW=2 WD=21 split=77 LD=44 LL=6 from the
champion's side — strong tail asymmetry). 95% CI [−3.6, +75.6]: not
conclusively significant, but verdict B's null (d16 + RCT/QAT: −6.5 ± 12.3)
is now explained as recipe-masked — plain deeper labels help, the RCT/QAT
recipe threw the gain away. 600-game extension running
(sprt/av-d16bc-vs-s1-ext600.*) to settle significance.

## 2026-10-17 (later) — d16 extension + xattn gate: weak-positive d16, conditioning null

**d16 extension (600 games):** d16bc +12.2 ± 14.2 (pair model, engine-name
recount; fastchess txt +12.17 ± 16.70 agrees). Combined with the first 300:
**d16bc ≈ +20 ± 11.5 over 900 games (1.75σ)** — the sign held in both
halves, but this is below the adoption bar. Status: weak-positive, unresolved;
one more ~900-game block would cross ~2.5σ if the effect is real.

**xattn head gate (loop 5):** EXACT tie with the pooled head — 220/2000 =
11.0% puzzles for both (same 20M records, steps, batch, init). Move
conditioning is NOT the bottleneck at this data scale. Loop 5 closes as a
measured null; loop 4 (encoding/trunk) inherits the 65-point gap question.
Noted for loop 4: DeepNoc's compute-matched 20K-game point saw ~10M samples
(one pass) vs our 4.6M samples over 20M records (0.23 epochs) — a recipe
gap that must be controlled before encoding surgery is credited.

## 2026-10-18 — d16 label-depth: definitive NULL after 1800 games

Third block (900 games): d16bc −4.7 ± 11.9. Sequence across blocks:
+36.0 ± 20.2 (300) → +12.2 ± 14.2 (600) → −4.7 ± 11.9 (900). Pooled:
**+7.7 ± 8.3 Elo over 1800 games (0.93σ)** — a null. The +36 was a false
positive, caught by the extension rule exactly as designed. Lesson recorded:
a 1.78σ result is NOT a "live lever" — with multiple comparisons in flight,
only ~2.5σ+ extensions justify adoption. Verdict B's null stands on clean
evidence now: deeper SF labels (d10 → d16) do not move strength through
plain BC at 1-2M positions.

## 2026-10-18 (later) — compute-matched control: samples are NOT the hidden variable

26k steps / 9.2M samples (2x the av_s6 run, same records/head/batch/init):
**12.45% puzzles vs 11.0%** — +1.45 points per doubling of samples. Nearly
flat scaling, vs DeepNoc's ~76% at the same compute point. Combined with
loop 5's exact head tie, the 65-point gap is now established by elimination
as structural in the trunk. Prime suspect: our BC warm-start (features
shaped for best-move classification, not win-prob regression; DeepNoc
trains from scratch). Next: from-scratch run, same 20M records/12k steps
-- the last cheap control before encoding/trunk surgery.

## 2026-10-19 — warm-start null + the underfitting reframe (loops 3-5 corrected)

From-scratch AV run (same 20M records, 12k steps, batch, pooled head):
**11.00% puzzles — exact tie with the warm-started run.** Warm-start is
exonerated.

The reframe: bucket-acc was STILL IMPROVING at run end (7.3% -> 10.9%
across 12k steps), loss 3.72 is far from floor, and 11-12% TRAIN-bucket
accuracy on deterministic SF-search labels = severe underfitting. The
"flat sample scaling" of the 26k control was an artifact of stopping an
unconverged run, not evidence about data or architecture. DeepNoc's 9M
recipe consumes ~30-100M samples; ours got 4.6-9.2M. The compute-matched
claim in the 2026-10-18 entry was wrong — the controls eliminated budget,
not architecture.

Next: 60k-step convergence run (~23M samples, ~28h). Verdict rule: if the
curve is still rising at 60k, the route is budget-bound (loop 4 closes as
"out of compute"); if it plateaus well below DeepNoc's 9M curve, the gap
is structural (encoding/trunk surgery justified).

## 2026-10-21 — loop 4 closed: the AV route is budget-bound (scaling curve complete)

Game-level point for the 60k-step net (23M samples): **28.3% vs champion,
−164 ± 28 Elo** (300 games, pair model). With puzzles 14.6%, the curve is:

| samples | puzzles | play vs champion |
|---|---|---|
| 4.6M | 11.0% | −225 ± 38 |
| 23M | 14.6% | −164 ± 28 |

Both metrics rise together: the DeepNoc AV recipe works, directionally
confirmed at game level, but at +2.2 puzzle points per sample-octave the
route needs ~30-100M+ samples (3-10 GPU-days) to reach even their BC
baseline's territory. All four cheap hypotheses are measured
(head conditioning, sample-doubling within budget, warm-start, budget) and
exactly one — budget — is binding. The 65-point gap to DeepNoc's curve at
matched param count is compute, not architecture.

En route: two wrapper bugs fixed (style scoping, hook-handle clobber) that
the old smoke test missed; the smoke test now covers position+go+bestmove.

## 2026-10-21 — loop 7: INT8/QAT net is the NEW CHAMPION (+136 ± 18.7, 7.3σ)

The stale int8_sprt chain (armed Oct 4, never fired) is now executed.
Two matches:

1. **bc/net_q_v2.bin (INT8, 8.2 MB) vs bc/net_e1.bin (FP32, 25.8 MB):**
   q_v2 scores **79.25%**, +233 ± 30 Elo (400 games; PGN recount by engine
   name — fastchess's txt Elo line was MISLABELED here, the recount caught
   it; direct PGN count agrees with the pair model).
2. **bc/net_q_v2.bin vs the champion av_v3/net_e1.bin:** q_v2 scores
   **68.63%, +136.0 ± 18.7 (7.3σ)** (400 games, pair model).

**New champion: /home/wyatt/data/runs/bc/net_q_v2.bin** — 3.1× smaller AND
+136 Elo over the old champion under tc=15+0.2.

Caveats recorded honestly:
- **Provenance RESOLVED (2026-10-23)**: the quantizer was never lost —
  `trainer/export_int8.py` (calibrated PTQ, exact centered-madd integer
  path) and `train.py --qat` (quantized-projected SGD) exist since commit
  0b9795b (Oct 4). net_q ≡ net_q_v2 ≡ INT8 of bc/net_e1.pt (cos 1.0000).
- **INT8-bonus verdict (2026-10-24): REAL but OLD-NET-SPECIFIC — NOT a
  general lever.** The replication battery landed:
  * Re-export through export_int8 reproduces the champion's s8 weights
    43/43 bit-exactly; only activation-calibration scales differ (few %).
    Champion-calib vs fresh-calib blobs tie in games (−12.2 ± 24.6, 100
    pairs) — the +233 is calibration-insensitive.
  * OLD net: FP32 blob loses to its own INT8 by −251.9 ± 31.3 (fresh
    calib) — replicates the original +233.
  * NEW net (bc40m): FP32 BEATS its own calibrated INT8 by +147.2 ± 26.8
    (100 pairs). The bonus INVERTS on a stronger net.
  * Mechanism findings: lonw_sim.py's parser was missing the per-Linear
    biases (fixed; net_e1.bin now bit-exact vs net_e1.pt, and
    engine-FP32 == lonw_sim == torch 60/60 on midgame positions — the
    FP32 engine path is CORRECT). An earlier 20%-agreement reading was a
    PROBE BUG (the 3-way probe never sent `setoption WeightsFile`, so the
    engine played netless). Engine-FP32 vs engine-INT8 agree 57/60 on
    midgame moves; the old net's +250 from ~5% policy flips stands as a
    real but idiosyncratic quantization-noise effect on a brittle net.
  * ALSO: bc40m's INT8 export FAILS the NetQ parity gate (5/50) — open
    defect in export_int8 on the new weight distribution; investigate
    before shipping any INT8 of the new champion.
- **TC-depth decomposition: RESOLVED, no confound.** Both matches ran at the
  default `RecyclePasses=1` — a single trunk forward per move for both
  engines regardless of speed (uci.cpp:65). Wall-clock speed cannot buy
  depth in a one-pass engine, so the +136 is pure evaluation quality. (A
  RecyclePasses>1 comparison would reintroduce the depth axis; not run.)
- The "RCT/QAT refuted" verdicts were on the av_v3 lineage (d16 fine-tunes);
  this bc-lineage QAT net is a different animal and is now the strongest
  measured net.

## 2026-10-22 — path-forward sweep: RP axis null; absolute Elo anchored (~1900 SF-limit)

**RecyclePasses sweep (the engine's own depth axis, never before measured):**
RP2 vs RP1 +5.2 ± 24.6, RP4 vs RP1 +13.9 ± 24.6 (RP4 perspective), RP8 vs
RP1 +3.5 ± 24.6 — all null at 200 games each. The searchless champion is at
its plateau; looped-trunk compute does not convert to strength.

**External calibration (first absolute anchors of the program):**
vs SF UCI_LimitStrength (200 games each, pair model):
- SF-Elo 2000: champ −129.0 ± 26.3
- SF-Elo 2200: champ −193.2 ± 28.5
- SF-Elo 2400: champ −251.9 ± 31.3

The SF limit-strength scale is compressed (~3x) in this range; interpolating,
the champion sits around **~1900 SF-limit-Elo** (roughly CCRL-blitz ~1900-2000
equivalent). Every future "+N" now has an absolute meaning.

**BC-at-scale launched:** bag_to_shard.py converts ChessBench AV bags to
engine shards (BC target = SF move, white-POV wdl from mover-POV win prob).
First conversion had a rank-orientation bug (FEN starts at a8; boards were
flipped, 99.5% of targets illegal, the first 400-step "run" was void) —
caught by reconstructing FENs from shard records vs source; fixed and
verified 100000/100000 board-exact. 37.9M-record shard converted; 74k-step
1-epoch BC run training (fast mask path, ~6h). Question it answers: does
20x the champion's data at the champion's architecture move the ladder?

## 2026-10-24 — CORRECTED: CHAMPION RETAINS; bc40m ARM INVALID (exposure-confounded)
**CORRECTION of an earlier entry written this day** (which claimed "+405.5 new
champion"): that reading inverted the recount attribution. The recount script
prints `engine1 (White of game 1) = <name>` BEFORE the counts; in the chain4
ladder and its rerun engine1 was **champ**, so +405.5/+424.3 were CHAMP's Elo
over bc40m, not bc40m's over champ. Lesson codified: never read a recount
without its attribution line (do not `tail -3` past it).

True, fully consistent measurement set (all PGN-recount, attribution-checked):
- **champ (net_q_v2) +405.5 ± 35.3 over bc40m** (150 pairs) and +424.3 ± 64.0
  in an independent rerun (50 pairs). Champ is +330.2 ± 51.7 over bc23m-fp32.
- **bc23m-fp32 +117.2 ± 26.0 over bc40m**: the 37.9M ChessBench control is
  WORSE than the old 23M-era net's FP32 form at equal play conditions.
- **bc40m −798 ± 175 vs SF-UCI-2000** (50 pairs, score 1%), and 5.8% on the
  local 500-puzzle suite (old net: 46.2%). Puzzle diagnosis was CORRECT —
  bc40m is simply a weak net; the "games vs puzzles" inversion never existed.
- **GAB-v3-lite −499.7 ± 44.6 vs champ** ✓ (as originally reported).
- **INT8-old +251.9 ± 31.3 over FP32-old** (fresh calibration; calibration-
  insensitive: champ vs fresh-calib −12.2 ± 24.6). bc40m INT8 −147.2 ± 26.8
  vs its own FP32 (normal PTQ damage; its export also fails parity 5/50).
- Netless-engine baseline: −177.2 ± 39.4 vs bc23m-fp32 (useful floor).

Why the bc40m arm failed — **exposure confound, not a data verdict**: bc_v1
trained 150M positions over 3 epochs; bc_v2 (champ parent) ran multi-epoch on
~122M-record shards; the bc40m control saw a single pass of 37.9M records
(38M samples). The arm is ~3-12× undertrained relative to the lineage it was
laddered against. Loop 1's data question remains OPEN at matched exposure.

Also this day (verified, stands):
- lonw_sim.py parser fixed (was skipping per-Linear biases → all tensors after
  L0.Wq were misaligned). net_e1.bin now bit-exact vs net_e1.pt, and
  engine-FP32 == lonw_sim == torch 60/60 on midgame positions: the engine's
  FP32 path is correct. (An earlier 20%-agreement reading was a probe bug —
  the 3-way probe never sent `setoption WeightsFile`, so the engine played
  netless.)
- ChessBench AV shards 4–15 downloaded (12 × ~1.3GB) and a 16-bag ~150M-record
  shard conversion produced. NOT queued for training until the exposure
  question is settled — next GPU slot goes to a matched-exposure control.

## 2026-10-25 — SHARD AUDIT + MATCHED-EXPOSURE VERDICT (chain6) + A/B (wdl fix)
- **chain6 (old pool @ bc40m's exact exposure: 98,750 steps × 384, fp32):
  v2pool +424.3 ± 52.3 over bc40m** (75 pairs, score 92%) and **v2pool
  −82.6 ± 29.2 vs champ** — the old pool at ONE pass of 38M samples nearly
  reaches the champion (which had 150M×3-class exposure + the INT8 effect).
  At matched exposure the ChessBench pool is ~424 Elo behind the old pool.
- **Shard audit (chessbench_bc40m)**: targets legal 300/300, mask sidecar
  complete (37,924,824) and covers targets 300/300, conventions consistent
  (a1=0, black pt+8), boards 100% unique, memorization probe 100%/256
  records. The data is CONSISTENT — not scrambled.
- **Own-shard fit**: bc40m net matches only **4.2%** of its own training
  targets (chance ≈ 3%) vs v2pool **52.9%** on its own. The difference is
  conditional target entropy: bc_v2's targets are consistent/predictable
  (self-play-era style); ChessBench AV targets are deep-Stockfish single
  moves — one pass gives too little signal per sample.
- **A/B on 2M-record prefixes (3k steps)**: as-is vs wdl-fixed
  (wdl=(pw,0,0) → (pw,(1−pw)/2,(1−pw)/2)). Fix cures the value collapse
  (loss 0.86–1.0 vs 0.0000) but the policy stall is UNCHANGED (match
  ~0.035 both). The `(pw,0,0)` bug was real (value head trained to
  always-win; engine ignores value) but is not the policy story.
  bag_to_shard.py now writes proper triples for future conversions.
- **Exposure test launched (chain7/watcher)**: bc40m continued +2 epochs
  (114M total samples) from net_e0.pt; ladder vs bc40m / v2pool / champ
  fires on training exit. Decides whether ChessBench closes the gap at
  3× exposure before any 150M-record commitment.
- **behavioral_cloning_data.bag (34GB, DeepMind) downloading** — GM-move
  targets (DeepNoc BC-9M: 65.7% puzzles), the natural low-entropy BC pool
  for our trainer if the exposure test leaves a gap.

## 2026-10-30 — DEEPNOC-PUZZLE CROSS-LINEAGE BENCHMARK (10k public suite)
Our nets on DeepMind's published ChessBench puzzle CSV (10,000 puzzles,
SF-best-move matching, converted to jsonl for score_puzzles):

| net | pool / exposure | accuracy |
|---|---|---|
| v2pool | old lichess pool, 38M samples | **43.0%** |
| net_e1 (champ parent) | same pool, 450M × 3 | **42.3%** |
| bcpool (human-move BC) | BC pool, 38M samples | **34.7%** |
| bc40m | ChessBench-AV, 38M | **10.4%** |
| (published: DeepNoc BC-9M) | BC bag, 670M positions | 65.7% |
| (published: DeepNoc GC-9M) | AV bag + value-max | 88.9% |
| (published: Leela-CF-policy) | 3.25M GM games | 93.5% |

Findings:
1. **Puzzle accuracy saturates at ~43% by 38M samples** — the champ's extra
   450M×3 exposure adds ZERO puzzle accuracy while carrying +83 Elo.
   Puzzle accuracy ≠ strength, now measured within-lineage.
2. **At equal Elo (bcpool ≈ v2pool within noise), the human-move net scores
   8 points lower on tactical puzzles** — puzzle suites are biased strength
   proxies across data types (the law extends across distributions).
3. The AV-data net's 10.4% on SF-best-move puzzles confirms the target-type
   diagnosis on DeepNoc's own benchmark.
4. Our best nets sit at ~2/3 of DeepNoc's BC-9M with 1/17 of the data —
   consistent with per-sample parity and a data-scale gap (the marquee
   tests exactly this).
Converted suite: /home/wyatt/data/chessbench/puzzles_deepnoc.jsonl.

## 2026-10-30/31 — SF-Q EVR MEASUREMENT: SF-agreement anti-correlates with strength
(trainer/evr_sfq.py; 334 positions sampled from slice0 [40M,145M]; SF at
800k nodes; per position: SF before-eval (mover POV) + per-net after-eval
(opponent POV); cp_lost = cp_before − cp_after_mover)

| net | mean cp lost | median | game-Elo order (ladders) |
|---|---|---|---|
| bcpool (38M human-move BC) | **0.9** | 5.0 | weakest of trio (−49 vs v2pool) |
| v2pool (38M old pool) | 10.8 | 9.0 | middle |
| net_e1 (champ parent, 450M×3) | 15.9 | 13.5 | strongest FP32 |
| bc40m (ChessBench-AV) | 196.9 | 159.0 | weakest overall |

**COMPLETE ANTI-CORRELATION**: the more SF-like the moves, the weaker the
net in game ladders — across all four. This is the third falsified strength
proxy (after validation loss and puzzle accuracy): **SF-move-agreement is a
style metric, not a strength metric.** Only paired games measure strength.

Confounds, honestly: bcpool's positions ⊂ its training data (in-distribution
reproduction deflates its cp-lost); net_e1 vs v2pool is the clean pair (both
out-of-distribution on slice0: v2pool closer to SF yet ~94 Elo weaker).
bc40m is the sanity anchor (worst everywhere ✓).

Tool: trainer/evr_sfq.py (reusable; any net, any shard range, any node
count). The before-evals are shared across nets; per-net after-evals
dominate the cost.

## 2026-11-01 — MARQUEE VERDICT: the 450M human-move pass FAILS
The marquee (4 chained slices, 450M human-move records, --init-from chained)
completed. Gate 1 ladder (250 games, attribution checked):

| match | result |
|---|---|
| **champ vs bcpool450** | **champ +226 ± 27** (125 pairs, 78.6% — bcpool450 wins ZERO games) |
| **sf2000 vs bcpool450** | sf2000 **+506 ± 67** (68 pairs, 95% for sf2000) |
| champ vs bc40m (AV, for reference) | champ +405 (already known) |

**bcpool450 is decisively weaker than every prior net.** The 450M human-move
pass made the net WORSE, not better. The scaling hypothesis (more human-move
data → stronger) is **FALSIFIED** at this exposure level.

Mechanism (measured):
- The BC pool's match rate climbs to 0.54 (human-move consistency) but this
  optimizes imitation of human moves, not winning games
- The net develops perfect mirror symmetry (0.0000) but that's a
  representation property, not a playing-strength property
- cp-lost to SF at 800k nodes: bcpool 0.9 (most SF-like = weakest),
  net_e1 15.9 (most deviant = strongest) — **anti-correlation confirmed**

**The deep conclusion: human-move BC has a hard ceiling.** The approach
saturates at ~38M samples and more data makes it worse (the net converges
to the human-move distribution's mean, which is drawish and passive).
The 526M BC pool cannot push past 2000 by scaling.

**The only proven path to higher strength remains the deep-SF-annotated
data (AV bags) with the AV head architecture and value-max serve** —
DeepNoc's AC-line recipe (their 9M model at ~2100-2200). Our AV
infrastructure exists (4 bags converted, the AV trainer works) but the AV
data has the same unlearnability cliff at low exposure — the fix is the
Stack (BC trunk + AV head) or more AV data at higher exposure.

The marquee verdict falsifies the "just add more human games" branch of
the roadmap and redirects all compute to the AV/stack path.

## 2026-11-02 — COMPOUND VERDICT: the 450M+distill+mirror compound FAILS
The compound (450M human-move records + --distill1ply 16 + --rank-mirror,
4 chained slices, batch 256 fp16) completed. Ladder results:

| match | result |
|---|---|
| **champ vs compound** | **champ +263 ± 29** (125 pairs, compound scores 18%) |
| **sf2000 vs compound** | sf2000 **+382 ± 46** (80 pairs, compound scores 10%) |

The compound is WORSE than bcpool450 without distill (champ +226 over
bcpool450 vs champ +263 over compound) — the distill + mirror additions
are NEUTRAL-TO-NEGATIVE at the 450M scale.

**The BC pool ceiling is now triple-confirmed:**
1. bcpool (38M BC): champ +424 → BC pool loses badly
2. bcpool450 (450M BC, no distill): champ +226 → 4× data doesn't help
3. compound (450M BC + distill + mirror): champ +263 → distill/mirror don't help either

The root cause: the BC pool's neutral value targets (⅓⅓⅓) provide no
value-gradient signal to the trunk, weakening the representation for the
policy head. The old pool's REAL game results (wl targets from actual
game outcomes) train the value head properly, which shapes the shared
trunk representation for the policy head too.

**The path forward requires real value targets.** The AV bags' win_prob
(deep-SF assessment) is the strongest available source. The AV head
architecture + the marquee trunk = the Stack experiment (queued).

## 2026-11-03 — THE STACK: AV head + value-max serve FALSIFIED at 37.9M scale
The Stack experiment (RESEARCH-CROSSFIELD §10): BC trunk from the marquee
(450M human moves) + AV head trained on 37.9M SF-annotated records + play
by AV-head argmax expected-win (the Python AV UCI wrapper at TC 15+0.2).

Result: the Python AV wrapper loses **-478 ± 73** to the champ policy
(50 pairs, 100 games, the AV wrapper wins only 6% of games).

Mechanism: the AV head at 37.9M records is not a reliable move-quality
evaluator. Its puzzle accuracy (16.75% vs policy 10.4%) showed promise,
but puzzle sharpness ≠ game strength (the recurring pattern). The AV head
needs ~670M+ records (DeepNoc's BC-9M scale) to produce a reliable
move-quality signal — at our scale it's still noise.

Implication: the AV head direction requires the full 1.1TB AV bag dataset
(2144 shards, 1.1TB storage) and the AV head architecture at scale —
beyond our current compute. The path to 2500 is:
1. The champ recipe (real game results + multi-epoch) — proven, ~1900
2. Policy-head QAT + GM-move fine-tune — the DeepNoc delta, +100-200
3. The AV head at scale — needs infrastructure beyond our budget
The champion (net_q_v2) remains the strongest net at every measured scale.

## 2026-11-04 — Soft SF-policy A/B: TIE; target-match is the 4th falsified proxy
The sharp-vs-learnable experiment (RESEARCH-CROSSFIELD): identical 84,746
BC records, identical exposure (20 epochs, batch 256, lr 3e-4, fp32);
the ONLY variable is the policy target — SF multipv=5 softmax(cp/120)
soft distribution (arm soft) vs the single hard SF move (arm hard).

Training: arm soft matched its targets 7x better (match 0.24 at epoch 12,
still climbing, vs the hard arm's 0.034 plateau — the unlearnability wall).
Ladders (150 games each, attribution-checked):
  soft vs hard:  soft -27.9 +- 28.5  (statistical TIE)
  soft vs champ: champ +432.3 +- 53.3 (placement; both tiny-data nets far below)
  (deepnoc puzzles for these nets: not measured — the ladder verdict is the metric)

FINDING: target-match (learnability) does NOT convert to game strength.
The 4th falsified proxy: validation loss, puzzle accuracy, SF-agreement,
and now trainability — no single-position metric predicts strength.

Mechanism: tau=120 spreads mass nearly uniformly (0.22/0.20/0.20/0.18/0.18);
the soft arm learns "these 5 moves are similar" and plays near-randomly
among them; the hard arm cannot predict WHICH move, so it plays noise.
Equal noise, equal strength. The policy-target SHAPE is not the bottleneck
at 84k records. Also note: the slice0 records carry NEUTRAL value targets
(value loss pinned at ln3 in both arms) — a clean isolation: policy-target
shape alone adds ~nothing.

The remaining delta between the BC pool and champ is the VALUE TARGET:
bc40m carries binary SF win_prob (compound38m value loss ~0.28), the old
pool carried real game results. The decisive experiment is value-realism,
not policy-softness. Compound38m (with binary value targets + distill +
mirror) fires ~02:00; if it still loses to champ by ~+250, the value
void is confirmed as the whole gap, and the path is: restore the old
pool (lichess archive) or attach AV-bag win_prob targets to BC records.

## 2026-11-05 — compound38m: distill+mirror ANTAGONISTIC (−97); mirror falsified as an addition
The 38M compound (distill 16 + rank-mirror together, binary value targets,
98,750 steps batch 384 fp32) vs ladders, attribution-checked 150 games:
  compound38m vs bcpool: bcpool +97.4 +- 29.5  (the compound LOSES to control)
  compound38m vs champ:  champ +458.5 +- 56.9  (consistent with bcpool -424)

The method matrix at 38M is now complete:
  mirror alone:        +16   +- 28   tie
  distill alone:      +147   +- 31   the ONLY positive method
  distill + mirror:    -97   +- 29   ANTAGONISTIC interaction

The rank-mirror erases the distill's gain and then some. Mechanism
hypothesis: the mirror halves true-distribution exposure while the
distill loss needs exposure to converge; or the distill's afterstate
self-consistency is broken under the rank+color transform. Either way:
the BC-pool recipe going forward is DISTILL ONLY, no mirror.

Note on the 23:58 ladders: the original run's fastchess invocations
completed instantly with no output (cause unresolved — consistent with
a dev-tree flap); re-run 02:08-02:24 produced clean 150-game ladders.

Program state: every training-method axis at 38M is now measured. The
remaining untested delta vs champ is VALUE-TARGET REALISM (real game
results vs binary SF win_prob): bc40m carries binary targets and still
lands ~-424; the old pool (deleted, restorable from the lichess
archive) carried real results and produced the champ. Next: restore
the old pool or attach AV-bag win_prob value targets (ternary, with
draws) to BC records, retrain bcpool+distill on it, ladder vs champ.

## 2026-11-05 (02:40) — bcpooldist DIRECT vs champ: -325; old-pool restore launched
Direct 150-game ladder (the earlier +330 was an estimate from ladder
arithmetic): champ +325.2 +- 41.7 over bcpooldist. The measured ladder:

  champ                          0
  v2pool (38M old pool)        -139 (est; verify direct next)
  bcpooldist (BC + distill)    -325 +- 42  (direct)
  bcpool (BC)                  -424 (direct)
  compound38m (BC+dist+mirr)   -459 +- 57  (direct)

Consistent with distill +147 over bcpool. The dominant delta is the DATA
SOURCE (~+285 Elo: real game results vs SF-annotated BC pool, equal scale
and exposure). Restoring the old pool: the lichess 2026-07/08 archives
(29+30 GB) SURVIVED the cleanup — ops/oldpool_restore.sh (dbb129f) re-shards
(min-elo 2000, 40M/month), combines to bc_v2_combined (~130M real-result
records), trains the bcpool_distill recipe (98,750 steps + Loop B distill),
ladders vs champ AND vs bcpooldist (the cross-pool distill transfer test).
Disk: bc_full shard (33.7 GB, falsified direction, no mask) deleted;
bc_sub slice masks deleted; 47 GB free.
NOTE: fastchess APPENDS to an existing -pgnout file — the Nov-1
bcpooldist-vs-champ.pgn mixed 300 games; recount used the -1105 split
(2026-11-05 GameStartTime only). Always split or delete before rerunning.

## 2026-11-05 (15:00) — THE ANCHOR CORRECTION: v2pool is -275, not -139; the champ's margin is SCALE x EXPOSURE
The restored old pool (80M records from the surviving lichess 2026-07/08
archives, min-elo 2000, min-ply 8, max-ply 120) trained with the
bcpool_distill recipe (98,750 steps, batch 384, fp32, Loop B distill).
Three direct 150-game ladders, attribution-checked:

  oldpool_distill vs champ:      champ +301 +- 40  (15.0% points)
  oldpool_distill vs bcpooldist: oldpool +220 +- 34 (78.0% points)
  v2pool vs champ (DIRECT):      champ +276 +- 38  (17.0% points)
  oldpool_distill vs v2pool:     TIE  -2.3 +- 28   (50.3% points)

CORRECTIONS TO THE LEDGER:
1. v2pool = -139 was an ESTIMATE and wrong; the direct value is -276.
   The "data-source delta +285" derived from it collapses to ~+40-60
   (bcpooldist 13.3% vs v2pool 17.0% points vs champ).
2. The distill's +147 (BC pool) does NOT show on the old pool:
   oldpool_distill = v2pool exactly (50.3%).
3. Elo intransitivity note: champ-vs-bcpooldist 86.7% vs
   oldpool-vs-bcpooldist 78% looked inconsistent in Elo because the
   pair-model diverges on extremely lopsided scores; at game level all
   four measurements are mutually consistent.

THE REAL STRUCTURE (game-level, mutually consistent):
  champ (150M x 3 epochs)     ~85% vs every 1-exposure net
  v2pool = oldpool_distill    (17% / 15% vs champ) - distill adds ~0 here
  bcpooldist                  13.3%
  bcpool                      ~5%
The champ's ~280-Elo margin is SCALE x EXPOSURE (450M samples vs 38M),
not the data source, not the training method, not the target shape.

NEXT (the replication, launched): bc_v2_combined (80M) x 3 epochs + distill
= 625k steps at batch 384 (~55h GPU) = 53% of the champ's sample budget on
the restored pool. If it lands champ+-50, the lever stack (more months +
epochs + distill) is validated and the 2500 path is data acquisition.

## 2026-11-08 (03:30) — replication crashed on ENOSPC; restarted from 0; disk hardened
The 80M x 3-epoch replication died at 84% (epoch 2 step ~560k): ENOSPC at
22:39 Nov 7 — the box's disk churn (an unrelated session ballooning ~85GB
then releasing it) filled /home mid-training. Two crash-restart cycles; the
script-level guard skips only COMPLETED runs (net_e2_ema.bin present), so
the restart began a fresh 3-epoch run — ~22h of compute lost. The run is
now at ~49k steps; verdict moves to ~Nov 10 late evening.

Disk hardening (76G free):
- bc_sub 450M slices deleted (32G du — falsified direction; soft100k shard kept)
- big_2026-07/08 part shards deleted (4.8G — bc_v2_combined supersedes)
- all *.pt intermediates + stack_av checkpoints deleted (runs/ 9.8G -> 8.1G);
  every ladder-referenced net kept (champ, bcpool, bcpooldist, v2pool,
  oldpool_distill, compound38m, soft_ab finals, replication epochs)
LESSON: on this box, any multi-day run needs ~30GB of disk headroom beyond
its own writes, and scripts that cannot resume must be restart-safe by
construction (skip-guard on the FINAL artifact) — which this one was; the
gap was that the crash was external (ENOSPC), so no artifact existed.

## 2026-11-11 (04:40) — THE QUANTIZATION CONFOUND: ~200-280 of every "gap to champ" was serving speed
The replication (80M x 3 epochs + distill) finished Nov 10 21:34. Its
verdict ladders + the precision controls that followed:

  champ (net_q_v2, INT8)  vs repl_fp32:      champ +297 +- 39  (repl 15.3%)
  champ (INT8)            vs repl_q (INT8):  champ +108 +- 30  (repl_q 35.0% WINS 46% of games!)
  repl_q (INT8)           vs repl_fp32:      repl_q +280 +- 38  (83.3% points)
  repl_q (INT8)           vs champ_fp32:     repl_q +489 +- 61  (94.3%!!)

CONCLUSION: serving precision is worth ~200-280 Elo at fixed TC. The champ
is INT8 (net_q_v2); every historical challenger ladder ran FP32 .bin nets.
The entire "gap to champ" ladder table (bcpool -424, v2pool -276,
oldpool_distill -301, replication -297) mixed INT8 speed against FP32
slowness. TRUE ladder requires INT8-normalized measurements.

INT8-normalized so far: champ 0, repl_q -108 — the replication is the
#2 net ever trained, within 108 of the champion, and its INT8 form beats
the champ's own FP32 parent by 489. The replication training DID work.

Multi-epoch note: repl_q = v2pool (1 epoch) is still TIE at game level
in the FP32 ladders — the epoch effect remains unproven either way;
every epoch/multi-epoch claim needs INT8-normalized re-measurement.

METHOD LAW (new, #5): the serving format is part of the measurement.
Ladders are only valid at matched precision. Every keeper net gets
exported to INT8 (.pt retained for re-export); every future ladder is
INT8-vs-INT8 (or FP32-vs-FP32).

Caveat: the earlier "INT8 bonus +179-233 (old-net-specific)" was this
same effect measured on the champ; it did not generalize to BC-era nets
then — because the bonus needs a STRONG net to amplify. It replicates
at +280 on the strongest FP32 net we have.

## 2026-11-11 (05:00) — THE INT8-NORMALIZED LADDER: transitive at last; epochs matter, precision-poisoning confirmed
All family nets converted to INT8 (bin2pt.py round-trip verified
byte-identical) and re-laddered at matched precision, 150 games each:

  champ_q            0      (reference)
  repl_q          -108 +- 30   (80M x 3 epochs + distill)
  oldpool_q       -211 +- 30   (80M window, 1 epoch + distill)
  v2pool_q        -218 +- 30   (38M old pool, 1 epoch)
  v2pool_q        -12  +- 28   vs oldpool_q (TIE — the two 1-epoch nets equal)

PERFECT TRANSITIVITY: 108 + 110 = 218 ✓. The FP32 ladders' intransitivity
was the precision confound. CORRECTED LEDGER FINDINGS:
1. EPOCHS MATTER: repl_q is +105-110 over both 1-epoch nets — the FP32
   "repl = v2pool tie" was poisoned. Two extra epochs = ~+105.
2. The FP32-era "oldpool_distill = v2pool tie" holds at INT8 too (-12 tie).
3. repl_q vs repl_fp32 = +280: the serving-precision bonus is net-specific
   (champ's own was +179-233). Stronger nets amplify the bonus more.
4. Every future claim of "gap to champ" MUST be INT8-normalized.

Also: runs/bcpool_distill/net_e0_ema.bin vanished post-ladder (~05:00,
likely an invisible-window event reaching the data dir) — all other
load-bearing nets audited OK. Retrain queued (resume-capable), then the
INT8-normalized distill + data-source deltas (bcpool_q vs v2pool_q, and
bcpooldist_q vs repl_q) complete the corrected ladder.

## 2026-11-12 (04:00) — THE DISTILL INVERSION: +147 at FP32, -131 at INT8 (matched precision)
The completed INT8-normalized ladder (all nets exported via bin2pt.py +
export_int8.py, 150 games each, attribution-checked):

  champ_q            0      reference (net_q_v2)
  repl_q          -108 +- 30   80M x 3 epochs + distill
  oldpool_q       -211 +- 30   80M window, 1 epoch + distill
  v2pool_q        -218 +- 30   38M old pool, 1 epoch
  bcpool_q        -274 (via v2pool_q +56: measured directly -56 +- 29)
  bcpooldist_q    -395 (via bcpool_q -131 +- 30 DIRECT: bcpooldist_q 32%)

FINDINGS:
1. Data-source delta (real results vs BC annotations, 38M x 1ep):
   v2pool_q - bcpool_q = +56 +- 29. Real, modest, matches the Nov 5
   estimate of +50.
2. THE DISTILL INVERSION: the Loop B distill is +147 +- 31 at FP32
   serving (Nov 3) and -131 +- 30 at INT8 serving (today) on the SAME
   pool/exposure. The effect SIGN depends on serving precision. The
   distill net's weights (V2 head max|w| 6.8 vs 3.5) quantize badly;
   INT8 per-tensor rounding erases the fine distinctions the distill
   training relies on. Its INT8 policy argmax lands on a degenerate
   Grob preference (g2g4 at startpos vs d2d4 for bcpool).
3. Recipe selection must happen AT SERVING PRECISION. FP32 ladders
   select the wrong recipe for INT8-served engines and vice versa.
4. The replication (80M x 3ep + distill) remains the best INT8-served
   non-champ net (-108): distill's INT8 harm is pool-dependent (it does
   not hurt the old pool at scale — repl_q beats every 1-epoch net).
OPEN QUESTION: is repl_q's distill actually helping at INT8? The clean
test is repl-no-distill (80M x 3ep without --distill1ply) exported INT8
vs repl_q — queued as the next GPU run (55h). If no-distill ties or
beats, drop the distill from the INT8 recipe.

## 2026-11-14 (09:40) — REPLICATION ARC CLOSED: the corrected ladder complete; the distill is optional
The no-distill control finished (80M x 3 epochs, no distill, INT8-served):
  repl_nodist_q vs repl_q:    TIE  -13.9 +- 28  (52.0% — direct, decisive)
  champ_q vs repl_nodist_q:   champ +170 +- 32  (repl_nodist_q 27.3%)

THE COMPLETE INT8-NORMALIZED LADDER (the corrected ledger, all direct):
  champ_q          0      20M x 2ep @ batch 512 (Sep-26 lineage)
  repl_q        -108 +- 30   80M x 3ep + distill
  repl_nodist_q -170 +- 32   80M x 3ep, no distill
  oldpool_q     -211 +- 30   80M window, 1ep + distill
  v2pool_q      -218 +- 30   38M old pool, 1ep
  bcpool_q      -274 (derived; direct -56 vs v2pool_q)
  bcpooldist_q  -395 (derived; direct -131 vs bcpool_q)

FINAL DISTILL VERDICT: OPTIONAL at INT8 on the real pool (direct tie
-14 +- 28; indirect via champ +60 — suggestive, not significant; keep it,
it costs nothing and the direct measurement shows no harm). The BC-pool
inversion (+147 FP32 / -131 INT8) stands as the precision-sensitivity
warning but does not generalize.

THE QUANTIFIED LEVER STACK for the 2500 push:
1. DATA SCALE: the champ's remaining ~108-170 edge tracks its 150M pool
   vs our 80M (450M vs 240M samples). Each lichess month = ~40M records.
   Scale to 150M-240M -> expect parity or better.
2. 3 EPOCHS: proven +105-170 over 1-epoch at matched precision.
3. INT8 SERVING: mandatory (law #5); .pt retained for every keeper.
4. Distill: optional; keep (neutral at scale, +147-class on BC-scale pools).
5. Then the DeepNoc deltas: policy-head QAT + GM-move fine-tune.
INFRASTRUCTURE (validated this week): step-ckpt + resume trainer, bin2pt
converter, flap-tolerant engine rebuilds, disk guards, INT8-first ladders.
NEXT: download 2-4 more lichess months -> 150M-240M pool -> train the
full-scale recipe -> expect champ parity -> public measurement (CCRL).

## 2026-11-14 vs 11-19 — SIGN AUDIT + THE CHAMPION IS DETHRONED
The scale160 verdict ladders (Nov 19) exposed sign errors in the ledger's
vs-champ entries: the sprt_pentanomial Elo is ENGINE1's (the White-of-
game-1 attribution line names it), and fastchess's Games summary is the
FIRST-command engine's — reading "Elo -150" as champ-positive instead of
champ-negative flipped three entries. Re-verified against the unambiguous
Games summaries (both perspectives recorded):

  scale160_q vs champ_q: scale160_q 70.33%  -> scale160_q +150 +- 31
  repl_nodist_q vs champ_q: repl_nodist_q 72.67% -> repl_nodist_q +170 +- 32
  repl_q vs champ_q (Nov 11): repl_q 65.00% -> repl_q +108 (ledger had it flipped)

THE TRUE INT8-NORMALIZED LADDER (vs champ = net_q_v2):
  repl_nodist_q  +170   80M x 3ep, NO distill  <- THE NEW CHAMPION
  scale160_q     +150   160M x 4-months x 3ep + distill
  repl_q         +108   80M x 3ep + distill
  champ (old)      0    20M x 2ep @ batch 512 (Sep-26 lineage)
  oldpool_q      ~ +5   1 epoch + distill   (TIE with old champ)
  v2pool_q       ~ -2   1 epoch             (TIE with old champ)
  bcpool_q       ~ -58
  bcpooldist_q   ~ -189

CORRECTED STORY:
1. The old champ was NEVER strongest at matched precision — the Sep-26
   recipe loses ~+100-170 to everything trained since at 3 epochs.
2. THE SIMPLEST RECIPE WINS: 80M x 3ep, no distill (+170) beats the
   doubled-data scale-up (+150) — sample scaling SATURATED between
   240M and 480M samples (2026-05/06 data adds no strength).
3. Distill on the big pool: slightly negative (170 vs 150 indirect;
   direct tie -14) — consistent with the BC-pool inversion. Drop it.
4. CHAMPION DECIDER running: repl_nodist_q vs scale160_q direct.

## 2026-11-19 (13:15) — CHAMPION DECIDER: repl_nodist_q stands; sample scaling SATURATED
  scale160_q vs repl_nodist_q: TIE  +16.2 +- 28  (scale160_q 47.7% points)

The doubled-budget run (480M samples, 160M 4-month pool) does NOT beat the
240M-sample repl_nodist (80M x 3ep, no distill). Sample scaling saturated
between 240M and 480M samples. THE CHAMPION: repl_nodist_q — the simplest
recipe (80M real-result pool, 3 epochs, no distill, INT8-served), +170
over the old champ, half the compute of scale160.

THE CORRECTED FINAL LADDER (INT8-normalized, direct where measured):
  repl_nodist_q  +170 +- 32   NEW CHAMPION (80M x 3ep, no distill)
  scale160_q     +150 +- 31   (tie with repl_nodist: +16 +- 28 direct)
  repl_q         +108 (65% vs champ)
  champ (old)      0
  oldpool_q       ~ +5
  v2pool_q        ~ -2
  bcpool_q        ~ -58
  bcpooldist_q    ~ -189

WHERE THE 2500 PATH GOES NOW (levers by measured yield):
1. DONE: real-result pool (+56), 3 epochs (+105-170), INT8 serving (+280).
2. SATURATED: sample count beyond 240M (+0). Data acquisition is NOT the
   next lever.
3. NEXT LEVERS (the DeepNoc deltas, untried): policy-head QAT at training
   (INT8-native weights), GM-move fine-tune on repl_nodist. Then the
   public measurement (CCRL) for ground truth.

## 2026-11-19/20 — THE SEARCH PATH: four bugs fixed; the blocker is now eval SPEED (~50-100ms)
Phase 0 of the search roadmap executed. The AB searcher had FOUR stacked
bugs — fixing them one at a time changed the failure mode each time:

1. uci: 'position startpos moves ...' never applied the moves list (the
   startpos branch didn't consume the 'moves' keyword; apply_moves then
   failed on the literal token and the engine answered the startpos root).
   The fen path consumed it correctly — why fen-based ladder games were
   always legal. THIS also misdiagnosed the AB known-bug TODO: ab() was
   searching the startpos, not misreading a position (ab6e723).
2. search: them = Color(~uint8_t) — bitwise-NOT promotes to -1/-2, not
   XOR-1: invalid Color fed attacked()/king_sq()/occupancy() (phantom
   in_check, OOB reads) (ba4212f).
3. search: is_draw returned reps >= 1 — the rep_stack always contains the
   node's own zobrist, so EVERY child node scored as a draw -> the whole
   search evaluated everything as 0 -> move choice = first legal move
   (a5a527c). This alone explains the AB arm's -478/-524/-489 ladder
   disasters (passive pawn pushes, zero wins).
4. uci: ABNodes parsed with substr(vpos+9) — chopped '400000' to '0000'
   -> node_cap 0 -> time_up everywhere (ab6e723). With node_cap fixed the
   search finds mate-in-1 instantly (validated: Qa8# on the back-rank FEN).

REMAINING BLOCKER (measured): NetQ::evaluate ~50-100ms per call (~8-20
evals/s) — the 3s move budget fits only ~25-30 evals = depth 1 + a few
children. The corrected AB-vs-searchless ladder still loses -489 BECAUSE
the AB arm times out mid-root-move and plays the first legal move.
Perft is perfect at depth 4+ (apply/movegen exonerated). The eval cost:
~25M scalar MACs (O(64x64xhd) attention x 8 layers) + per-call input
requantization + the full 4096-entry policy head computed per leaf eval
when the search only needs the 3-float wdl.

THE OPTIMIZATION PROJECT (next session, est. 20-50x total):
a. value-only fast path: skip policy scores entirely for non-root nodes
b. multithread the per-head attention + per-square FFN loops (6 cores)
c. fuse quantization into the producing step (kill per-call lrint loops)
d. optional: softmax/exp vectorization
Target: <=3ms/eval -> 1000+ evals per 3s move -> depth 3-4 + qsearch ->
REAL search strength over the champion net. That is the 2500 path:
search over a trained eval is how every strong engine in history got there.

## 2026-11-20 (01:10) — ABSOLUTE ELO CALIBRATED: repl_nodist_q = ~1940 (SF-anchored); the 2500 gap is ~560, and the biggest lever is a WORKING SEARCH
Calibration ladder vs Stockfish 16.1 UCI_Elo 2000 (UCI_LimitStrength,
tc 15+0.2, 50 games, openings.epd): repl_nodist_q scored 42.0% (16 W / 24 L)
-> SF-anchored absolute Elo ~1940 +- 50.

THE 2500 GAP: ~560 Elo. Lever ranking by measured/expected yield:
1. WORKING SEARCH (+300-600 class): the AB searcher is now logically
   correct (four bugs fixed: startpos-moves handler, ~uint8_t Color,
   is_draw >= 1, node_cap parse) and finds mates — but it starves:
   NetQ::evaluate costs 122-174ms (value-only vs full), so the 3s move
   budget fits ~25 evals = depth 1. Target <=10ms/eval via: value-only
   path (done, -30%), SIMD/vectorized attention (O(64x64xhd) scalar
   loops), quantization fusion into layernorm outputs, threads when the
   box is idle. A depth 4-6 search + qsearch over this eval typically
   clears 2500.
2. Eval-quality deltas (+30-100 each): elite fine-tune (min-elo 2500
   shard from the on-disk archives), policy QAT (--qat, untested at scale).
3. Saturated: sample scaling (480M = 240M).
ALSO MEASURED: NetQ::evaluate throughput ~8-20 evals/s single-thread —
the engine's INT8 eval is ~100x slower than NNUE-class implementations;
per-node net evals with a 2048-batched deadline check made the AB arm
play moves instantly-mislabeled (the "connection stalls" failure).

## 2026-11-20 (06:00) — THE AB SEARCH WORKS; THE VALUE HEAD IS THE CEILING (human-noise-calibrated)
The t60 ladder (8s/move, 400k nodes, tc 60+1, 50 games): the AB arm lost
91% again. But the per-move forensics flip the diagnosis: the AB arm's
SEARCHED moves (8s each ✓ verified by move timings) were reasonable
(Kc3 attacking the b2 rook, Kc2); the LOSING moves were the POLICY arm's
instant tactical shots (Rb1/Rxf1/a1=R). And the decisive experiment on
the exact position: at depth 1 the search correctly finds Kxb2 winning
a rook; at depth 2 it re-scores Kxb2 the same as quiet retreats.

DIAGNOSIS: the value head is CALIBRATED TO HUMAN NOISE. It was trained
on raw lichess game outcomes — in human games, rook-up positions get
drawn and blundered constantly, so the trained win-probability for a
rook-up position is only ~56-65%, not 85%. value_cp therefore prices
a captured rook at ~+120, indistinguishable from a quiet retreat. THE
SEARCH CANNOT ADD STRENGTH OVER NOISY VALUE TARGETS. The policy head
stays sharp because "moves humans played" generalizes; the value head's
targets do not.

MEASURED CALIBRATION (SF-anchored, Nov 20): repl_nodist_q = ~1940 +- 50
(42% vs SF@2000). The 2500 gap is ~560. Eval speed: value-only eval
~122ms (the threading under the box's load gives little; the single-
thread cost is ~25M scalar MACs + 524k scalar lrint requantizations +
524k scalar erff gelu calls per eval).

THE FIX (the Lc0 thesis, applied): retrain the VALUE HEAD on DEEP-ENGINE
win probabilities instead of raw results. The searchless_chess AV bags
carry (position, move, win_prob) with win_prob from deep Stockfish —
2 bags on disk (2.9GB), more downloadable. Value target: win_prob from
the AV bags; game results elsewhere; then the AB search over a CLEAN
value head adds real strength (this is exactly how Lc0-class engines
get strength: search over policy + clean value).
NEXT SESSION: download/inspect more AV bags -> trainer --value-sf
targets -> retrain repl_nodist with SF-anchored value targets ->
the AB-vs-searchless ladder decides whether search unlocks 2500.

## 2026-11-20 (17:00) — fast-math null; the selftest confirms the value head is materially insensitive; the DeepNoc-recipe pivot
- -ffast-math on netq.cpp: no measurable speed change (125/107ms value/full
  under box load) — GCC did not auto-vectorize these loop shapes. The real
  eval-speed fix needs hand-written AVX2 (attention dots, requant, gelu) —
  a dedicated project, parked behind the searchless line.
- THE SELFTEST (new instrument-gate binary, src/selftest.cpp): perft ✓,
  AB-legality ✓, mate-in-1 ✗ at 200k nodes (✓ at 500k — budget), and the
  EVAL-SIGN BATTERY: the value head rates WHITE UP A FULL QUEEN (winning
  position, 26 legal moves) at wdl0 = 0.572 — materially insensitive.
  THE HUMAN-NOISE DIAGNOSIS CONFIRMED BY THE INSTRUMENT: the value head
  trained on raw lichess outcomes cannot price material.
- THE STRATEGIC PIVOT: the searchless line first (the DeepNoc recipe:
  elite GM-game data + policy QAT — DeepNoc hit 2374 SEARCHLESS). The
  search track waits for the eval-speed project + the SF-value retarget
  verdict (ft-sfvalue running: value loss 0.85 → the retarget in progress,
  value-head-only, trunk+policy frozen — zero-drift A/B design).
