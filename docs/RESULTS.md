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
