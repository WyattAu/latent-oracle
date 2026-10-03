# Research Notes — Architectures

**Scope:** architecture-level findings from the 2024–2026 literature, evaluated
for a 6.4M-param encoder that plays searchless chess.

---

## 1. Chessformer — Geometric Attention Bias (GAB)

**Paper:** "Chessformer: A Unified Architecture for Chess Modeling" (May 2026)
**Claim:** a single encoder-only transformer advances SOTA on all three chess
modeling goals (strength, human-move prediction 57.1% move-matching,
interpretability).

### 1.1 Design
- Squares as tokens (like ours), encoder-only (like ours).
- **Geometric Attention Bias:** attention logits get a learned bias derived
  from the geometric relation between query square and key square:
  `A_ij += b[rel(i, j)]` where `rel` buckets the square-pair relation
  (same rank/file/diagonal distance bands, knight offsets, king adjacency).
  Unlike a static relative-position table, GAB is *dynamic* — the bias is
  computed from piece-aware geometry and adapts to the position.
- Attention-based source-destination policy head (structurally the same as
  our `E_from · E_to / √d` head — independent validation of our choice).

### 1.2 What we adopt
- Replace learned absolute positional embedding with a per-head learned
  **relation bias table**: `b[head][bucket(i,j)]`, buckets =
  {dist 0, dist 1, dist 2, dist 3+, knight-offset, same-file, same-rank,
  same-diag}. 8 buckets × 8 heads = 64 bias scalars — negligible params.
- Keep our SD policy head as-is (validated by the paper).

### 1.3 Expected gain
Chess paper reports gains at 270M scale; at 6.4M the bias mainly helps
attention form attack/defense structure in layer 1-2 (we observed our model
learns this from scratch at cost of capacity). Estimate +20–50 Elo.

---

## 2. Mixture of Masters (persona MoE)

**Paper:** "Mixture of Masters: Sparse Chess Language Models with Player
Routing" (Feb 2026)

### 2.1 Design
Small GPT experts each trained on one world-class grandmaster's games; a
post-hoc gating network picks the persona per move. Beats dense baselines
trained on aggregated data.

### 2.2 Verdict: DEFER (P4)
- We don't have per-GM corpora segmented by player at our license level;
  Lichess aggregation loses attribution granularity for clean-room use.
- The result is about **style diversity** (avoiding mode-averaging), not raw
  strength. Our bottleneck is strength.
- Relevant later: if we build an opening-repertoire product, persona experts
  (aggressive/solid) become a feature. Note as future work.

---

## 3. AlphaViT family

**Paper:** "AlphaViT" (2024). Vision transformers inside AlphaZero: encoder
only (AlphaViT), encoder+decoder (AlphaViD), decoder with action embeddings
(AlphaVDA). One network plays multiple games / board sizes.

### 3.1 Verdict: partial adopt
- Confirms encoder-only + transformer value head works for full game play
  with MCTS; our setting removes MCTS but keeps the trunk.
- Variable board size is a product feature we don't need (standard 8×8 only).
- Nothing new to implement; cite as supporting evidence for trunk design.

---

## 4. Lookahead Attention

**Paper:** "Autoregressive Modeling with Lookahead Attention" (May 2023)

### 4.1 Design
At each step, the model extrapolates K hypothetical continuations of the
context using a cheap proposal (the model itself), appends them to the
sequence, and attends over the extended string. The final prediction is
conditioned on "what would happen if…" — search implemented as attention
over imagined futures.

### 4.2 Verdict: PILOT (P3)
- Conceptually the bridge between our BC model and DiffuSearch: same
  "imagine futures, then decide" structure, but inside one architecture.
- Cost: K forward passes over imagined continuations per move. At our scale
  (64-token sequences, 6.4M params) K=4 costs ~5× latency — acceptable for
  analysis mode, not for 60+6 blitz.
- Risk: proposal quality. Our policy is only ~2000 Elo; imagined continuations
  will be noisy. DiffuSearch sidesteps this with trained denoising; lookahead
  attention does not. Try only after AV predictor exists (needs a value signal
  to rank imagined futures).

---

## 5. Looped transformers (recurrent depth)

**Papers:** "Decoding Looped Transformers Better for (Almost) Free" (Oct 2026,
introduces **LoopCD**), "Scaling Laws for Looped Mixture of Experts" (Sep
2026), "Looping Beyond Twice" (Oct 2026).

### 5.1 LoopCD (see RESEARCH-INFERENCE.md for inference detail)
Shared block executed R times; earlier passes give weaker predictions.
Contrastive decoding between final and an earlier pass improves selection —
training-free, and lets you halve R at equal quality.

### 5.2 Why this matters for us
Our recycling plan (run the trunk k times with feedback) makes the model a
*looped* transformer. LoopCD says: don't just average or take the last pass —
**contrast** passes. And the loop-scaling paper says gains persist as loops
scale. Adopt LoopCD alongside recycling from day one (it's ~40 lines of
inference code).

---

## 6. Prior session findings that still stand

| Finding | Source | Status |
|---|---|---|
| 270M params + AV labels = 2895 Elo searchless | Ruoss et al. 2024 | adopted as target recipe |
| Lc0 internally represents future moves (92% at 2 plies) | look-ahead interpretability paper | justifies future-token formats |
| Tracking vs deciding data contradiction | 2026 dual-capability paper | explains our H1 negative |
| DiffuSearch +540 Elo via discrete diffusion | arXiv:2502.19805 | P2 pilot |
| Bidirectional attention required for diffusion | same | in SPEC-DIFFUSION.md |
