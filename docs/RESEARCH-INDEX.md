# Research Index: Latent-Oracle v0.3 Planning
| [RESEARCH-NOVEL.md](RESEARCH-NOVEL.md) | Invented: HiCo history-conditioning, Recycle-Consistent Training, UGAR gate; with prior-art checks and kill criteria |
| [RESEARCH-NOVEL2.md](RESEARCH-NOVEL2.md) | Invented: CTAP child-token attention (amortized 1-ply search), AMZ amortized minimax (oracle-ceiling breaker), LID spec; pre-registered experiment ladders |
| [SPEC-BLOB-V3.md](SPEC-BLOB-V3.md) | Single bundled format change: castling/ep, material buckets, rating, HiCo, GAB-in-NetQ; validation checklist |
| 1 | Action-value (AV) prediction target | Ruoss et al. 2024 (ChessBench) | **ADOPT**; highest impact | P0 |
| 6 | Warm start + opponent curriculum + keep-best | Gold-standard lightweight study (Jul 2026) | **ADOPT**; validated across games | P1 |
| 7 | DiffuSearch denoising inference | arXiv:2502.19805 | **PILOT**; biggest single gain but new paradigm | P2 |
| 8 | Lookahead attention | arXiv 2305 (2023) | **PILOT**; elegant, unproven at our scale | P3 |
| 9 | Mixture of Masters (persona MoE) | arXiv 2602 (Feb 2026) | **DEFER**; style, not strength; needs per-GM data | P4 |
| 10 | JEPA world model / test-time training | JEPA-TTT (Sep 2026) | **WATCH**; needs dynamics model we don't have | P4 |
| 11 | Multi-token prediction heads | MTP post-training (Oct 2026) | **WATCH**; gains mostly in token-space LMs | P4 |
| 12 | Imitation learning / DAgger | Gold-standard study | **REJECT**; confirmed unhelpful for lightweight agents | - |
| 13 | Extra capacity to break ceiling | Gold-standard study | **REJECT**; capacity does not break data ceiling; matches our H1 | - |
6. **RL-GRPO stage**; IMPLEMENTED (grpo_train.py: Gumbel-top-K groups,
   (oracle-ceiling breaker); runs on BC-v1 weights as soon as the GPU