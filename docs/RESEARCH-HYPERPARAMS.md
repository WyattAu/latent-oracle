# Hyperparameter Recommendations — latent-oracle 6.4M

**Model:** 8 layers, d=256, 8 heads, FFN 1024, policy emb 128, ~6.4M params.
**Hardware:** RTX 2060 6GB, batch 512, ~50M-position shards.
**Basis:** Llama-line best practices, Muon literature, scaling-law checks,
our own SPRT-measured baselines. Each entry states current → recommended.

---

## 1. Architecture

| Param | Current | Recommended | Why |
|---|---|---|---|
| Norm | LayerNorm (pre) | **RMSNorm (pre)** | equal or better, ~0 cost, standard in Llama 2/3 |
| Activation | GELU | **SwiGLU** (FFN 1024 → gate 683+683 or keep 1024 fused) | better Pareto at same params |
| Positional | learned absolute 64 | **relation bias table** (GAB-lite, 8 buckets × 8 heads) | Chessformer validation; chess-native geometry |
| QK-norm | none | **add** (RMSNorm on q,k per head) | attention stability, free |
| Attention bias | none | keep none | validated at scale |
| Value head | mean-pool → MLP | keep | fine |
| Policy head | E_from·E_to/√d | keep | validated by Chessformer independently |

Params stay ~6.4M (SwiGLU reshuffle is param-neutral with d_ff 683).

## 2. Optimizer

| Param | Current | Recommended | Why |
|---|---|---|---|
| Optimizer | AdamW | **Muon (hidden 2-D) + AdamW (embed/head)** | orthogonalized momentum converges faster |
| Muon LR | — | 0.02, momentum 0.95, Nesterov | standard small-model range |
| AdamW LR (embed/head) | 3e-4 | 6e-4 with Muon on hidden | Muon paper pairing |
| Weight decay | 0.01 | 0.1 (AdamW params only) | standard |
| β | (0.9, 0.95) | keep | fine |
| Grad clip | 1.0 | keep | fine |
| Precision | fp16 AMP | keep (bf16 if GPU allows; 2060 is fp16-only) | — |

## 3. Schedule

| Param | Current | Recommended | Why |
|---|---|---|---|
| Warmup | ~1% steps | **5%** of total steps | stability at lr 3e-4+ |
| Decay | none (const) | **cosine → 10% of peak** | standard; final-epoch export quality improves |
| Epochs | 3 | keep (see data ratio §4) | — |
| z-loss | none | add 1e-3 · (logsumexp)² on policy logits | logit drift protection for long fine-tune chains |

## 4. Data / batching

| Param | Current | Recommended | Why |
|---|---|---|---|
| Batch | 512 | keep | 6GB constraint; LR already tuned to it |
| Tokens/param/epoch | ~7.8 | 2.6 epochs ≈ compute-optimal (Chinchilla 20/param) | current 3 epochs fine |
| Shuffle | shard order | keep | shard = random games already |
| Policy target | hard top-1 | **decisiveness-weighted hard top-1** (implemented) | H2 test; upgrade to softmax-τ if format v2 lands |
| Value target | logistic(eval) for labeled | keep | consistent with SF WDL |
| Mask sidecar | 50M pre-generated | keep; generate for every new shard | fixes python-chess deadlock |

## 5. Fine-tune chain (validated pattern: warm start + keep-best)

| Stage | Init | LR | Epochs | Notes |
|---|---|---|---|---|
| BC pre-train | scratch | 3e-4 (Muon: 0.02) | 3 | running (bc_v1) |
| AV fine-tune | BC-v1 e2 | 1e-4 | 2–3 | labeled_5m.shard when ready |
| Quality fine-tune | AV ckpt | 5e-5 | 1 | quality filter on |
| Lc0 exposure | AV ckpt | 5e-5 | 1–2 | fine-tune, not from scratch |
| **Keep-best gate** | — | — | — | SPRT each export vs current best; only promote winners (gold-standard study §1.1.5) |

## 6. SPRT measurement defaults

| Param | Value |
|---|---|
| Pool opponent | SF16.1, plies 1 and 2, 60+6, 16MB hash |
| Openings | openings.epd (2000), random order, both colors |
| Games | 200/concurrency 5 per leg; 400 total for verdicts |
| Elo bounds | [-35, 35] H0 skill gap |

## 7. Explicitly rejected (evidence)

| Setting | Why rejected |
|---|---|
| d=384/10L (14M params) | data-undernourished at 50M; capacity doesn't break ceiling (gold-standard study) |
| Batch 1024 + 2×LR | OOM risk on 6GB; no expected gain at fixed epochs |
| Learned absolute pos + GAB together | redundant; GAB subsumes |
| Full DEQ/Anderson recycling | no gain at R≤3 (prior analysis, ADR-002) |
| Higher SF depth labels (16+) | H1: precision mismatch hurts at 6.4M params |
