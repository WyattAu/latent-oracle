# RESULTS: Measured Verdicts (living ledger)
## Chain 0: baselines (v0.1 era)
## H1: SF-label distillation (2026-09, v0.1 era)
## Chain 1: data scaling + mechanism fixes (2026-10-04/05)
| **H2** | identical labels, decisiveness-weighted loss | p1: **−280 ± 121** (24-game early stop) | **MECHANISM CONFIRMED**: +156 vs the −436 uniform baseline. SF labels aren't poison; uniform loss on them was. Still below BC; imitation precision gap persists (AV phase attacks it properly) |
## Mechanism baselines (E4): NOT cross-comparable (different opponents)
## Scope change: labeled set halved for time-to-verdict (2026-10-06)
saturates well before 4M; 2M x 2 epochs at batch 512 is 7.8k steps, and the
## Loss attribution: where the Elo actually goes (2026-10-06)
real but secondary; and the endgame is where tablebases already give exact
**Implication; this validates the armed plan rather than redirecting it:**
  material heuristics; together they are ~15-20% of the attributable losses
## MirrorAvg: inference-time test-time augmentation (2026-10-06, INCONCLUSIVE)
### CORRECTED (2026-10-08): the game verdict was read backwards: MirrorAvg HURTS
The uncomfortable part: two independent cheap measurements; the paired
eval delta (+23.4cp) and the v3 policy-accuracy delta (+1.3pp); both pointed
verdict is now 1.8 s.e. from zero; a real, cheap gain. MirrorAvg should be
**Decision: enabled; but deliberately not yet.** Flipping the default while
sample was noisy; the direction held.
## VERDICT A: CORRECTED: the AV bundle did NOT beat the BC chain (2026-10-07/08)
300 games, stage-1 listed first) reported **−18.55 ± 24.16** for stage 1 -
## VERDICT A: original misread entry (kept for the record): "the AV bundle works: +4 to +13" (2026-10-07)
First mechanism-level verdict of the project. The AV stage-1 bundle; v3
upsampling, mirror augmentation, EMA, Muon + WSD; fine-tuned on 2M d10
it; every prior full-mechanism attempt (distillation, DiffuSearch, AMZ)
stage-1 net; verdict B isolates whether the refinement adds anything.
## BC-v2: data scaling is exhausted in this regime (2026-10-07)
recipe on it; with one deliberate change: **1 epoch over 130M unique** shows
   data-quality ceiling for this corpus (Lichess games, SF-d10/d16 targets) -
## DiffuSearch: REJECTED by a strength test its own gate could not see (2026-10-06)
| **strength**; diffusion policy vs greedy play of the *same* BC weights | **0W 0D 12L** | playing strength |
| harness control; greedy BC vs greedy BC (identical policies) | 3W 2D 5L, 0.400 ± 0.158 | confirms the harness does not favour either side |
positions, both policies scored against the *same* reference; Stockfish's
~0.51 for BC-v1 while the first 1000 score 0.461; subset composition alone
fixed on the way; a duplicate-SEP off-by-one that shifted the a0 slot
## AMZ: REJECTED: oracle supervision destroys the policy (2026-10-06)
mismatch shows up exactly where it would; the policy learns to expect
`model(codes, side)`; it never passes castling, en-passant or rating, so those
| `king_bucket_emb.weight` | 1024 / 4096 | four rows; **live** (computed inside forward from the board) |
king-quadrant embedding ever contribute. There is no train/inference bug; the
contribution is zero on both sides; but the fields cannot earn Elo because
0.510; i.e. no gain, and the difference is under one standard error at this
## Volatility weighting: null at preview scale (2026-10-08)
+0.4pp; inside noise. Honest caveats: a single design point (W=2.0/150cp), a
Also fixed on the way: the flag silently never reached `build_batch`; the
## Verdict B: the d16 RCT/QAT refinement: +18.6 ± 24.2, the only upward game signal (2026-10-08)
perspective; the stage-2 net is ~+19 Elo up, within noise at 300 games
remaining ~165k records carry no TB rows; harmless, but a full-shard sidecar
## GRPO: two NaN sources fixed; the first real launch was destroyed by them
logged update and saved NaN weights (the later SPRT vs the SF pool; 0W/368L
- is simply what a NaN net scores; the promotion rule correctly refused it).
checkpoints DESPITE the NaN; "it exported" is not a finiteness check.
## GRPO: compute-blocked, handoff state (2026-10-08)
non-finite update skip, NaN-save refusal; see the entry above). The
learning question either; 120 steps at lr 1e-5 inside a 0.03 KL budget moves
| 0 | **DiffuSearch variants** (LID, CTAP, different horizons) | - | **dropped**: 23.3% policy accuracy vs SF-best (BC: 51.0%) and 0-12 in play; variants inherit the failure until that gap is explained |
| 0b | **AMZ variants** (coarser reply sampling, marginalizing the reply) | - | **dropped pending a mechanism fix**: reply-conditioned targets are privileged information a memoryless policy cannot use at inference |
| 1 | ~~Volatility weighting~~ | tested 2026-10-08: +0.4pp policy accuracy at preview scale (455 -> 459 / 1000); inside noise | kept behind `--volatility-weight` (default off); re-test at full corpus only if tactical robustness proves to be the binding constraint |
| 2 | **MirrorAvg: confirm at game level** | measured twice now; paired +23.4cp (CI [+5.3,+44.6]) and +1.3pp policy accuracy on a v3 net; free at inference | needs a clean SPRT; deliberately sequenced AFTER verdict A so it does not confound the AV comparison |
| 3 | **Policy-aware hard-example mining**; upweight positions where the *current* policy's greedy move loses >= 200cp vs SF | the direct version of volatility weighting on the model's own errors rather than the game's | requires one inference pass per labeled position; `mine_blindspots.py` already covers the value-head variant and has never been run on real moonshot output |
| OOM killed `lo-data label` at 3.9M/4M records; 17 h of labeling lost (no output until completion) | shared box ran another session's Lean/Go builds to 25+ GB | batching + `--resume` (a partial output is a valid input prefix); RAM preflight in all chain scripts |
| fastchess prints `Elo: X, nElo: Y` on **one line**, so `grep -oE 'Elo: ...'` also matches nElo; and `tail -1` selects exactly the wrong number. `analyze_verdicts` read a 6-game smoke match as −300.89 (nElo) instead of −190.85 | verdict-harness smoke run | `\bElo:` in Python; `sed 's/, nElo:.*//'` before extraction in `keep_best.sh` |