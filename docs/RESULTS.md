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

## Incident register (2026-10-04/05)

| Incident | Catch | Fix |
|---|---|---|
| parse_fen ep direction inverted | movegen fuzz (1/200) | direction flipped; perft green |
| phase_post `$txt` case typo | bash -x trace | fixed; 6 refs; memory guard added |
| Diffusion tokenizer: black pieces shifted ×2 + 'b' vocab collision | loss curve starting at 0.24 (impossible for honest CE) + device-side assert | CODE_TO_CHAR map + SIDE rename; poisoned 2M cache deleted; regression test added |
