# ADR-001: Adopt DiffuSearch — Discrete Diffusion for Implicit Search

**Status:** Accepted
**Date:** 2026-10-02
**Deciders:** Wyatt Au

## Context

Latent-Oracle is a searchless chess engine: given a board position, it
outputs the best move in a single forward pass with no tree search.

The fundamental limitation of this approach is that a single forward
pass cannot replicate the iterative exploration of a search tree.
Our BC-v0 baseline achieves ~2000 Elo (38% vs SF16-d1), but hits a
ceiling because one transformer pass cannot capture the tactical
calculation that search performs.

The DeepMind ChessBench paper (Ruoss et al. 2024, arXiv:2402.04494)
achieved 2895 Lichess blitz Elo with a 270M parameter transformer using
action-value prediction — but that model size is out of reach on our
compute budget (RTX 2060, 6GB).

## Decision

We adopt the **DiffuSearch** technique (arXiv:2502.19805, HKUNLP) as
the primary v0.3 architecture change. DiffuSearch achieves **+540 Elo**
over a same-size baseline by adding discrete diffusion over future
game states, enabling the model to "look into the future" through
iterative denoising rather than explicit tree search.

### What changes

| Component | Before | After |
|---|---|---|
| Attention | Causal (sees only past tokens) | **Bidirectional** (sees all tokens including future) |
| Input sequence | `state → move` | `state → move → future_state → future_move → ...` (horizon 4) |
| Training loss | Cross-entropy on move | Cross-entropy on randomly masked non-state tokens, weighted by 1/(t+1) |
| Inference | Single forward pass → argmax | **Progressive denoising**: T=64 steps, each unmasks ~1/(t+1) of masked tokens |

### What does NOT change

- Board core, movegen, perft, Syzgyg probe, time management
- Symbolic gates (repetition, 75-move, dead position)
- SPRT harness, opponent pool, evaluation methodology
- Data pipeline (shard worker, labeler — unchanged for human data)

## Rationale

1. **Proven result:** +540 Elo in the original paper with the same
   7M parameter GPT-2 architecture we use
2. **Same architecture family:** bidirectional GPT-2, not a new
   architecture type — minimal code change
3. **Code available:** github.com/HKUNLP/DiffuSearch (Apache-2.0)
4. **Directly applicable:** instantiated on chess, not adapted from
   another domain
5. **Our data is sufficient:** 50M BC positions + 3M Lc0 positions
   provide both current states and future trajectories

## Why alternatives were rejected

| Alternative | Why rejected |
|---|---|
| SF depth-16 label distillation | H1 proved it degrades play by ~309 Elo vs BC (search-precision mismatch) |
| Scaling model to 270M | Out of reach on RTX 2060 (6GB VRAM); would need cloud GPU budget |
| Pure BC with more data | Already at the BC ceiling (~2000); more data sharpens the same distribution |
| Lc0-style self-play RL | Requires self-play infrastructure we don't have; Lc0 took years to develop |

## Consequences

### Positive
- The model can "imagine" future positions through iterative denoising
- Bidirectional attention lets the current move attend to its own
  predicted consequences — a form of implicit search
- The +540 Elo gain is from a paper using the SAME architecture size
  and SAME training approach (supervised, not RL)

### Negative
- Inference cost increases: T=64 forward passes instead of 1
- The sequence length increases: 80 tokens per future depth × 4 depth
- The model must be retrained from scratch with the new format
- The INT8 quantization path must be redesigned for the diffusion
  inference pattern (denoising steps don't map 1:1 to single-pass INT8)

### Neutral
- The "discrete diffusion" framing is a theoretical justification;
  operationally it's BERT-style masked language modeling with
  progressive unmasking at inference time

## Implementation notes

See [SPEC-DIFFUSION.md](SPEC-DIFFUSION.md) for the full specification.

Reference implementation: github.com/HKUNLP/DiffuSearch (Apache-2.0)
The DiffuSearch codebase is based on LLaMA-Factory; our adaptation
extracts the core diffusion logic into a standalone implementation.
