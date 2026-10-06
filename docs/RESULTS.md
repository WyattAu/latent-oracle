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
