# Research Notes — Data: Augmentation, Curation, Features

**Scope:** getting more out of the 50M-position shard and the labeling
investment. Data is our binding constraint (H1, gold-standard study).

---

## 1. Board symmetry augmentation

### 1.1 What is legal in chess
Go/vision use 8 symmetries; chess pawn direction and castling kill most:
- **File mirror** (a↔h): legal. Pawns, castling, en passant all survive
  (rights/squares mirror consistently). This is the augmentation Lc0 uses.
- Color swap + rank flip: changes side-to-move; usable but our shards are
  side-annotated — doubles bookkeeping for 2× data.
- Rotations/reflections without color swap: illegal (pawns move one way).

**Verdict: file mirror = exactly 2× effective data for ~10 lines of code.**
Encode: mirror square `sq → sq ^ 7` (file flip), mirror move (u,v) →
(u^7, v^7), flip castling bits K↔Q sides, mirror ep file. Apply randomly
per record during training (prob 0.5) or double-pass the shard.

### 1.2 Interaction with GAB
Our GAB buckets are mirror-symmetric EXCEPT the learned bias table
(buckets are preserved under file mirror: same file → same file, knight →
knight...). File mirror maps bucket set to itself, so no GAB change needed.
Clean interaction. 

---

## 2. Curation: what to keep, what to drop

### 2.1 Already in place
- Elo ≥ 2000 filter at shard time (durable binary `--min-elo`).
- Blitz+rapid only (bizarre bullet moves excluded).
- `--quality-filter` in trainer (drop played-move-loses->300cp).
- Decisiveness weighting (soft-curation: down-weight rather than drop).

### 2.2 New candidates (ranked by expected value)
1. **Game-phase balancing.** Shards from long games over-represent
   middlegames; endings are underrepresented and are where searchless play
   is weakest (conversion failures → draw). Remedy: oversample records with
   ≤ 12 pieces by 2×, undersample move 1-8 openings slightly. Cheap shard
   re-ordering or trainer-side sampling weights.
2. **Opening dedup.** openings.epd SPRT book aside, training on 200K copies
   of the same 12 moves of theory wastes capacity. Cap per-ECO position
   families via Zobrist bucket count (cap 8 per position). Requires a
   shard pass; medium effort.
3. **Draw-aware value calibration.** Our logistic eval→WDL target (k=0.0037)
   is SF's model, not ours. After the AV fine-tune, fit k on OUR value
   head's predictions vs outcomes (Platt scaling). Improves the draw gates
   in engine.cpp (pwin>0.6 logic depends on calibrated WDL).
4. **Coreset selection / dataset distillation.** 2026 literature is
   image-centric; for supervised policy learning the closest cheap proxy is
   decisive+confident subsampling — which is exactly our quality filter +
   weighting. Verdict: no exotic coreset; the simple proxy is already built.

---

## 3. Input features beyond piece codes

Current tokens: piece code + square + side. Candidates from the literature:

| Feature | Evidence | Cost | Verdict |
|---|---|---|---|
| Attack/defend maps | Chessformer uses geometry bias (we have GAB); explicit attack channels help CNN-era nets | tokenizer rewrite, blob v3 | DEFER until GAB measured |
| Piece-count plane | 1 scalar, helps value head endgame conversion | trivial | ADOPT with next blob bump (v3) |
| Halfmove-clock plane | repetition awareness | trivial | marginal — engine has explicit repetition gate |
| Castling-rights one-hot | model currently infers from king/rook placement | trivial | ADOPT with v3 |
| En-passant file | rare but concrete | trivial | ADOPT with v3 |

Blob v3 = v2 + 3 extra input scalars → tokenizer change in both trainer and
engine. Bundle with the next architecture change (GAB measurement first,
then v3). **Do not churn format mid-experiment.**

---

## 4. Labeling economics (SF labels)

- 5M labels at depth 14 ≈ 7-14 h CPU (running). Depth scaling: depth 16
  would ~3× that for ~+30cp label quality; H1 showed label precision beyond
  the model's representable range is wasted. Depth 14 is the right budget.
- **Reuse labels across epochs**: labels live in the shard, so epochs are
  free. The marginal spend is one-time.
- Future: label the Lc0 3M self-play shard too (it lacks SF evals; those
  records currently serve BC-only). Adds 3M AV-training records for ~8 h
  CPU. Queue after 5M lands.

---

## 5. Openings book for play (not training)

- Current openings.epd (2000 positions) is fine for SPRT.
- For real play: a tiny 1-ply book from the training shard's most decisive
  openings (move = most common SF-best) avoids early-policy noise. Low
  priority; engine strength at move 1-6 is already acceptable via BC.
