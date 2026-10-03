# SPEC: Optimal Searchless Chess Model — Full Research Synthesis

**Status:** Design document for v0.3+
**Research sources:** 2024–2026 papers across ML, games, finance, protein folding, graphics
**Baseline:** latent-oracle BC-v0 (38% vs SF16-d1)

---

## 1. The Fundamental Limitation

A single forward pass cannot replicate iterative search. The gap between
one-pass evaluation and exhaustive tree search is the gap between ~2000
and ~3600 Elo. Every technique in this document attacks this gap from
a different angle.

## 2. Architecture (Optimal Design)

```
┌─────────────────────────────────────────────────────┐
│                  INPUT TOKENIZER                     │
│  64 squares × piece-type one-hot (13 classes)       │
│  + 2D positional encoding (rank × file RoPE)        │
│  + side-to-move plane                                │
│  + attack/defense/x-ray feature planes (optional)   │
├─────────────────────────────────────────────────────┤
│                  TRUNK                               │
│  L × [Bidirectional Self-Attention Block]            │
│      ┌─ LN → QKV Attention (RoPE, no bias)          │
│      ├─ LN → SwiGLU FFN                             │
│      └─ Residual + QK-norm                           │
│  Recycling: feed x back as input for R passes       │
│    (R = 1 for simple positions, R = 4 for complex)  │
├─────────────────────────────────────────────────────┤
│                  HEADS                               │
│  Policy: LN → E_from → dot E_to / √d → masked CE    │
│  Value:  LN → mean-pool → MLP → (W, D, L)           │
│  Piece-value: LN → MLP → per-piece eval (aux)       │
│  Variance: LN → MLP → per-move σ (risk-aware)       │
└─────────────────────────────────────────────────────┘
```

### 2.1 Key architectural choices (each backed by evidence)

| Choice | Evidence | Expected gain |
|---|---|---|
| Bidirectional attention (no causal mask) | DiffuSearch: future tokens enable implicit search | +540 Elo (paper) |
| Recycling (R passes) | AlphaFold, LoopCD (arXiv:2610.02185) | +50–150 |
| RoPE 2D positional encoding | Llama 3, ViT improvements | +20–50 |
| SwiGLU activation | Llama 2/3, Mistral | +10–30 |
| RMSNorm (not LayerNorm) | Llama 2/3, Mistral | +5–15 |
| QK-norm | Stability at scale | +5–10 |
| No attention bias | Llama 3 ablation | +0–5 |
| Lookahead attention | "Autoregressive Modeling with Lookahead Attention" (2023) | +30–80 |

### 2.2 What we deliberately DON'T do

| Rejected | Why |
|---|---|
| Causal attention | Blocks future-token attention (the core of DiffuSearch) |
| DEQ / Anderson | No gain at k≤3 scale; parked behind H5 trigger |
| MoE | Adds complexity; single model with recycling achieves similar effect |
| SDF king safety | Learned features dominate hand-crafted |
| Taylor Δ-caching | Side-to-mover flip; exact accumulators are cleaner |

## 3. Training Target: Action-Value Prediction

**This is the single highest-impact change** (DeepMind ChessBench ablation).

### 3.1 What changes

| | BC (current) | AV prediction (new) |
|---|---|---|
| Input | Position | Position + each legal move |
| Target | "Which move was played?" | "What is the outcome of this move?" |
| Loss | Cross-entropy on move index | Cross-entropy on WDL per move |
| Model learns | Imitate played moves | Evaluate each move |
| Generalization | Limited to training distribution | Generalizes to unseen positions |

### 3.2 Training data annotation

For each position in the training set, annotate every legal move with:
- SF16 WDL probability (win/draw/loss, 3 floats)
- SF16 eval in centipawns (1 float)
- Played move indicator (1 = was played, 0 = not played)

This requires SF analysis per position: ~200ms per position at depth 16.
For 1M positions: ~55 hours on 5 threads. For 5M: ~11 days.

Alternative: use Lc0's training_data/ binary format which has AV
annotations built in (from Lc0's own MCTS search). Format conversion
required but eliminates the SF labeling step.

### 3.3 Loss

```
For each position s in batch:
    For each legal move a in A(s):
        prediction = model(s, a)  → (P_win, P_draw, P_loss)
        target = SF WDL annotation for (s, a)
        loss += CE(prediction, target)

    played_move_loss = CE(played_move_logits, played_move_index)
    total = AV_loss + α × played_move_loss
```

The AV loss teaches the model to evaluate moves. The played-move loss
provides an additional signal (what humans actually play in practice).
α = 0.3 balances the two signals.

## 4. Training Pipeline

### 4.1 Phase 1: BC pre-training (BUILD FOUNDATION)
- Data: 50M human played moves (BC)
- Target: played move (CE) + game-result WDL
- Epochs: 3–5 until convergence
- Purpose: learn general chess patterns and representations
- This is ALREADY RUNNING (bc_v1_combined.shard)

### 4.2 Phase 2: AV fine-tuning (LEARN TO EVALUATE)
- Data: SF-labeled positions with per-move AV annotations
- Target: AV prediction for each legal move + played-move CE
- Learning rate: 1e-4 (lower than BC phase)
- Epochs: 2–3 until AV loss converges
- Purpose: shift from imitation to evaluation
- Base model: Phase 1 checkpoint

### 4.3 Phase 3: Quality fine-tuning (REMOVE NOISE)
- Data: quality-filtered subset (played move ≈ SF best)
- Target: same as Phase 2
- Learning rate: 5e-5 (even lower)
- Epochs: 1–2
- Purpose: remove residual noise from the policy
- Base model: Phase 2 checkpoint

### 4.4 Phase 4: Lc0 data fine-tuning (LEARN FROM 3600 ELO)
- Data: Lc0 self-play positions (3M from match PGNs)
- Target: Lc0 move distribution (if available in PGN format)
- Learning rate: 5e-5
- Purpose: expose the model to 3600-level move patterns
- Base model: Phase 3 checkpoint

### 4.5 Inference
1. Encode position
2. For each legal move: compute AV prediction
3. Select move with highest P(win) + 0.5 × P(draw)
4. Apply quality gates (Syzygy probe, repetition, 75-move)
5. Apply time management (low clock → PST fallback)

## 5. Data Requirements

| Phase | Data | Size | Source |
|---|---|---|---|
| BC pre-train | Played moves | 50M | Lichess Jul+Aug (✅ ready) |
| AV fine-tune | SF-annotated positions | 5M | Need SF labeling (~2 days) |
| Quality fine-tune | Filtered labeled | ~500K | From AV fine-tune data |
| Lc0 fine-tune | Lc0 self-play | 3M | match_pgns (✅ ready) |

## 6. Compute Budget

| Phase | GPU time (RTX 2060) | CPU time |
|---|---|---|
| BC pre-train (3 ep × 50M) | ~18 h | — |
| SF labeling (5M × depth 14) | — | ~3 days background |
| AV fine-tune (3 ep × 5M) | ~3 h | — |
| Quality fine-tune (1 ep × 500K) | ~30 min | — |
| Lc0 fine-tune (3 ep × 3M) | ~2 h | — |
| SPRT (3 models × 400 games) | — | ~6 h |
| **Total** | **~24 h GPU** | **~4 days background** |

## 7. Expected Results

| Model | Training | Expected Elo |
|---|---|---|
| BC-v0 (current) | 20M BC, 2 ep | ~2000 (measured: 38% vs SF-d1) |
| BC-v1 (50M data) | 50M BC, 3 ep | ~2050–2100 |
| + AV fine-tune (5M SF) | quality shift | ~2100–2200 |
| + Quality filter | noise reduction | ~2150–2250 |
| + Lc0 fine-tune | 3600-Elo exposure | ~2200–2300 |
| + Recycling | iterative refinement | +50–150 |
| + DiffuSearch inference | implicit search | +200–500 |
| **Total (all techniques)** | | **~2400–2800** |

The DiffuSearch inference (+540 in the paper) is the highest-impact
single technique. Combined with AV prediction and recycling, the
target of 2400+ is realistic with 6.4M parameters.

---

## 8. What We Deliberately Don't Do

| Rejected | Why |
|---|---|
| SF depth-16 BC targets | H1 negative: search precision mismatch |
| DEQ / Anderson | No gain at k≤3 |
| MoE | Recycling achieves similar with less complexity |
| SDF king safety | Learned features dominate |
| Taylor Δ-caching | Exact accumulators cleaner |
| Lattice crypto | Irrelevant to chess |

## 9. Deliverables

| Milestone | Deliverable |
|---|---|
| BC-v1 SPRT | Data-scaling Elo number |
| AV predictor | First searchless model with move evaluation |
| DiffuSearch inference | First model with implicit search via denoising |
| Quality filter | Ablation: filtered vs unfiltered training |
| Recycling | Ablation: k-pass vs single-pass |
| Full pipeline | Multi-phase training with all techniques |
| v1.0 | Complete research map with all results |
