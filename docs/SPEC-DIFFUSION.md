# SPEC: Discrete Diffusion Chess Policy (DiffuSearch Adaptation)

**Status:** Draft for implementation
**Implements:** ADR-001, ADR-004
**Reference:** arXiv:2502.19805 (DiffuSearch), github.com/HKUNLP/DiffuSearch

---

## 1. Overview

Replace the single-forward-pass policy with a discrete diffusion policy
that predicts future game states through iterative denoising. The model
takes the current position plus noise-corrupted future tokens and
progressively denoises them over T steps. The first denoised action is
the selected move.

## 2. Tokenization

### 2.1 Vocabulary

| Token type | Count | Encoding |
|---|---|---|
| Piece-square | 781 (13 × 64 - 51 impossible) | Each piece-color at each square |
| Move (UCI from-to) | 4096 (64 × 64) | From-square × 64 + to-square |
| Promotion | 4 (N, B, R, Q) | Per promotion direction |
| Special | 8 | PAD, MASK, BOS, SEP, Castle-K, Castle-Q, EP, NOISE |
| **Total** | **~6100** | |

### 2.2 Sequence layout

For horizon h=4, the full sequence is:

```
[SEP] [state_tokens(77)] [SEP] [a₀] [state_tokens(77)] [a₁] [state_tokens(77)] [a₂] [state_tokens(77)] [a₃]
 ^^^^ source (frozen)      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ target (diffused)
```

Where:
- `state_tokens(77)` = fixed-length FEN encoding (padded to 77 chars)
- `aᵢ` = the i-th action (UCI move token + optional promotion token)
- The source tokens encode the current position
- The target tokens encode: first action + future states and actions

During training, the source tokens are frozen (never masked).
The target tokens are randomly masked according to the diffusion
schedule. At inference, all target tokens start as MASK.

### 2.3 Token details

State tokens: each character of the padded FEN string maps to a
vocabulary entry. The FEN is normalized to exactly 77 characters:
- Piece placement: 64 chars (digits for empty runs expanded to dots)
- Side to move: 1 char
- Castling: 1 char (normalized: K=1, Q=2, k=4, q=8 mapped to chars)
- EP file: 1 char (or '-' if none)
- Padding: remaining chars filled with '.'

Future tokens: for each step j in the horizon:
- `a_j`: the move in UCI (from_square × 64 + to_square, plus optional promo)
- `s_{j+1}`: the resulting state (77-char FEN encoding)

Target tokens during diffusion: `a₀, s₁, a₁, s₂, a₂, s₃, a₃`
(horizon 4 = 4 state-action pairs = 8 target tokens + 3 state tokens = 11 tokens)

### 2.4 Masking schedule

At diffusion timestep t (of T total steps), each non-source token is
independently masked with probability (t+1)/T. Masked tokens are set
to a special [MASK] token in the vocabulary.

The source tokens are never masked. The model must reconstruct the
masked target tokens conditioned on the unmasked source tokens and
any already-unmasked target tokens.

## 3. Model

### 3.1 Architecture

A decoder-only transformer (GPT-2 style) with:
- **Bidirectional attention** (no causal mask)
- 8 layers, d_model=256, 8 heads, FFN=1024
- Total parameters: ~7M
- Vocabulary: ~6100 tokens
- Maximum sequence length: 77×5 + 8×4 = 417 tokens

### 3.2 Forward pass

```
input_ids → embed_tokens → transformer (bidirectional) → lm_head → logits
```

No diffusion-specific layers. The model architecture is identical to
the baseline transformer — the diffusion is entirely in the training
loss and inference procedure.

## 4. Training

### 4.1 Loss

At each training step:
1. Sample a random timestep t ∈ [0, T)
2. For each non-source token, with probability (t+1)/T, replace it
   with [MASK]
3. Forward pass the noised sequence
4. Compute cross-entropy loss on the masked positions only
5. Weight the loss by λ_t = 1/(t+1)

```python
loss = CE(logits[masked_positions], original_tokens[masked_positions])
loss = (loss * weight).sum() / num_masked
```

### 4.2 Data

Each training sample:
- Source: the current position (frozen, never masked)
- Target: the played move + the next `h-1` states and moves from the
  actual game continuation

For BC data: the target is the played move and the resulting states
from the game. For SF-labeled data: the target is SF's best move and
the resulting states from SF's PV line.

## 5. Inference

### 5.1 Progressive denoising

```
1. Initialize: source tokens from the position, all target tokens = [MASK]
2. For t = T-1, T-2, ..., 0:
   a. Forward pass → logits for each position
   b. For each masked token: sample from softmax(logits) → x̃₀
   c. Unmask ~1/(t+1) fraction of the least confident tokens
   d. Keep already-unmasked tokens unchanged
3. After T steps: extract a₀ from the first move token
```

The unmasking at each step uses the model's predicted probability to
select which tokens to unmask (easy-first decoding). More confident
predictions are unmasked earlier.

### 5.2 Move selection

After the final denoising step, the first move token (`a₀`) is the
selected move. If it's a promotion move, the promotion piece is taken
from the promotion token.

## 6. Evaluation gates

| Gate | Metric | Threshold |
|---|---|---|
| Token accuracy | First-move prediction accuracy | ≥ 40% (vs 20.83% for S-A baseline) |
| Puzzle accuracy | WAC puzzle solve rate | ≥ 30% |
| SPRT | Elo vs SF16-p1 and p2 | Report with ±error |
| Latency | Move time at batch=1 | ≤ 2 s (T=64 forward passes × ~30ms) |

## 7. Dependencies

- The board core (position parsing, movegen) for FEN encoding/decoding
- The training pipeline (mask sidecar, dataloader)
- The reference implementation: github.com/HKUNLP/DiffuSearch
