# Research Index — Latent-Oracle v0.3 Planning

**Last updated:** 2026-10-03
**Scope:** Full literature sweep for the optimal next model, expanding on
`SPEC-OPTIMAL-MODEL.md`. All findings cite papers; verdicts are project-specific.

---

## Document map

| Doc | Contents |
|---|---|
| [RESEARCH-ARCHITECTURES.md](RESEARCH-ARCHITECTURES.md) | Chessformer/GAB, Mixture of Masters, AlphaViT, lookahead attention, looped transformers |
| [RESEARCH-TRAINING.md](RESEARCH-TRAINING.md) | Lightweight-agent gold-standard study, decisiveness weighting, Muon optimizer, distillation, curriculum |
| [RESEARCH-INFERENCE.md](RESEARCH-INFERENCE.md) | LoopCD, test-time training, recycling, DiffuSearch inference |
| [RESEARCH-HYPERPARAMS.md](RESEARCH-HYPERPARAMS.md) | Concrete hyperparameter recommendations for our 6.4M model |
| [RESEARCH-OPTIMIZATION2.md](RESEARCH-OPTIMIZATION2.md) | EMA/SWA weight averaging, loss refinements, QAT, self-play fine-tuning, SAM |
| [RESEARCH-DATA.md](RESEARCH-DATA.md) | File-mirror augmentation, curation, input features, labeling economics |
| [RESEARCH-HPO.md](RESEARCH-HPO.md) | ASHA/Bayes search methodology, concrete budgeted HPO plan |
| [RESEARCH-EVAL.md](RESEARCH-EVAL.md) | SPRT power analysis, fast triage, proxy metrics, promotion policy |
| [RESEARCH-RL.md](RESEARCH-RL.md) | GRPO from SF rewards, Faynt precedent (10M + RL), anchored self-play, QTPT Q-targets |
| [RESEARCH-REPRESENTATIONS.md](RESEARCH-REPRESENTATIONS.md) | Distributional values, auxiliary tasks (JEPA-lite, reply prediction), trunk families |
| [RESEARCH-EXTERNAL.md](RESEARCH-EXTERNAL.md) | NNUE corpus (buckets, factorizer, SCReLU), KataGo efficiency, Lc0 production, Maia rating-conditioning |
| [RESEARCH-NOVEL.md](RESEARCH-NOVEL.md) | Invented: HiCo history-conditioning, Recycle-Consistent Training, UGAR gate — with prior-art checks and kill criteria |
| [RESEARCH-NOVEL2.md](RESEARCH-NOVEL2.md) | Invented: CTAP child-token attention (amortized 1-ply search), AMZ amortized minimax (oracle-ceiling breaker), LID spec — pre-registered experiment ladders |
| [RESEARCH-SYSTEMS.md](RESEARCH-SYSTEMS.md) | Measured latency (INT8 4.7×), optimization backlog, robustness inventory, statistical/testing rigor + gaps |
| [RESEARCH-ENDGAME-RL.md](RESEARCH-ENDGAME-RL.md) | Syzygy ground-truth endgame labels (free perfect labels), Gumbel-GRPO sampling, TD-consistency, conversion metrics |
| [SPEC-BLOB-V3.md](SPEC-BLOB-V3.md) | Single bundled format change: castling/ep, material buckets, rating, HiCo, GAB-in-NetQ — validation checklist |
| [SPEC-OPTIMAL-MODEL.md](SPEC-OPTIMAL-MODEL.md) | Earlier synthesis: AV targets, pipeline, compute budget |
| [SPEC-DIFFUSION.md](SPEC-DIFFUSION.md) | Diffusion tokenizer/training/inference spec |

## Verdict table (papers → actions)

| # | Technique | Source | Verdict | Priority |
|---|---|---|---|---|
| 1 | Action-value (AV) prediction target | Ruoss et al. 2024 (ChessBench) | **ADOPT** — highest impact | P0 |
| 2 | Decisiveness-weighted policy loss | Lc0 practice; H1 failure analysis | **ADOPTED** (implemented `3ce2f02`) | P0 done |
| 3 | Geometric Attention Bias (GAB) | Chessformer, arXiv 2605 (May 2026) | **IMPLEMENTED** (trainer `--gab` blob v2 + engine loader, commit b351076/38d315f) | P1 done |
| 4 | Recycling + LoopCD contrastive decoding | LoopCD, arXiv 2610.02185 (Oct 2026) | **IMPLEMENTED** (engine `RecyclePasses`/`LoopCDAlpha`, parity-verified vs Python) | P1 done |
| 5 | Muon optimizer (hidden) + AdamW (embed/head) | Muon line of work 2024–2026 | **IMPLEMENTED** (trainer `--optimizer muon`, AMP joint wrapper) | P1 done |
| 6 | Warm start + opponent curriculum + keep-best | Gold-standard lightweight study (Jul 2026) | **ADOPT** — validated across games | P1 |
| 7 | DiffuSearch denoising inference | arXiv:2502.19805 | **PILOT** — biggest single gain but new paradigm | P2 |
| 8 | Lookahead attention | arXiv 2305 (2023) | **PILOT** — elegant, unproven at our scale | P3 |
| 9 | Mixture of Masters (persona MoE) | arXiv 2602 (Feb 2026) | **DEFER** — style, not strength; needs per-GM data | P4 |
| 10 | JEPA world model / test-time training | JEPA-TTT (Sep 2026) | **WATCH** — needs dynamics model we don't have | P4 |
| 11 | Multi-token prediction heads | MTP post-training (Oct 2026) | **WATCH** — gains mostly in token-space LMs | P4 |
| 12 | Imitation learning / DAgger | Gold-standard study | **REJECT** — confirmed unhelpful for lightweight agents | — |
| 13 | Extra capacity to break ceiling | Gold-standard study | **REJECT** — capacity does not break data ceiling; matches our H1 | — |

## Current experiment queue (running)

| ID | Question | Status |
|---|---|---|
| BC-v1 | Does 50M > 20M human data? | training, epoch 0 |
| H2 (dw_retrain) | Does decisiveness weighting fix SF-label distillation? | queued after fine-tune |
| BC-v1f | Does quality-filtered Lc0 fine-tune help BC-v1? | chained in phase_post.sh |
| 5M labels | Raw material for AV target (P0) | labeling at depth 14 |

## Priority for next build cycle

1. **AV predictor** (P0): retrain on labeled_5m.shard with AV-style decisiveness
   weighting + two-hot eval targets + JEPA-lite/reply aux heads (RESEARCH-REPRESENTATIONS.md).
2. ~~GAB attention~~ **IMPLEMENTED** (trainer `--gab` + engine blob v2 loader).
3. ~~LoopCD at inference~~ **IMPLEMENTED** (engine `RecyclePasses`/`LoopCDAlpha`).
4. ~~Muon~~ **IMPLEMENTED** (trainer `--optimizer muon`).
5. ~~EMA weights + move-matching eval + file-mirror aug~~ **IMPLEMENTED**.
6. **RL-GRPO stage** — IMPLEMENTED (grpo_train.py: Gumbel-top-K groups,
   SF-reward pool, PPO clip + KL anchor, v3 inputs; grpo_phase.sh armed
   behind AV). Runs the moment the AV net exists.
7. **HPO stage A/B** per RESEARCH-HPO.md once the current chain lands.
8. **DiffuSearch pilot**: trainer+inference scaffold IMPLEMENTED
   (train_diffusion.py/infer_diffusion.py); GPU run queued behind the chain.
9. **Novel techniques** (RESEARCH-NOVEL.md/NOVEL2.md): RCT implemented
   (`--recycle/--rct-lambda`); AMZ offline pilot is the top research bet
   (oracle-ceiling breaker) — runs on BC-v1 weights as soon as the GPU
   queue drains; then CTAP with blob v3; HiCo after.
