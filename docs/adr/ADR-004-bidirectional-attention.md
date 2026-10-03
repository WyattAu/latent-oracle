# ADR-004: Bidirectional Attention for the Diffusion Policy

**Status:** Accepted
**Date:** 2026-10-02

## Context

DiffuSearch (ADR-001) requires the model to attend to future tokens
(planned moves and positions) during training. The current architecture
uses GPT-2-style causal attention, which masks all future positions.

The DiffuSearch reference implementation removes the causal mask:
```python
for gpt2block in self.model.transformer.h:
    gpt2block.attn.bias.fill_(True)  # full attention
```

## Decision

Use **bidirectional attention** (no causal mask) for the diffusion
model. The state tokens are frozen (not predicted); only the future
tokens are masked and denoised.

## Rationale

1. The model must attend to future tokens to predict the current move
   in context. Causal attention blocks this by design.
2. The state tokens (position) are always fully visible — they are the
   conditioning signal. Only the future tokens start masked.
3. Bidirectional attention allows information to flow between the
   current move and its predicted future consequences during training.
   This is what enables "implicit search."

## Consequences

- The model cannot be used for autoregressive generation (it sees the
  future). This is fine — inference uses progressive denoising, not
  autoregression.
- The attention computation is O(n²) over the full sequence, not O(n²/2)
  as with causal attention. At our sequence length (~80-160 tokens),
  this is negligible.
- The state tokens must be marked with a source mask so the loss only
  applies to future tokens.

## Implementation

In PyTorch, the GPT-2 attention module uses `bias` as a causal mask.
Setting it to `True` (all ones) removes the causal constraint. For our
implementation, we simply omit the causal mask in our custom transformer.
