# Research Notes: Architectures
## 1. Chessformer: Geometric Attention Bias (GAB)
  Unlike a static relative-position table, GAB is *dynamic*; the bias is
  our `E_from · E_to / √d` head; independent validation of our choice).
  same-diag}. 8 buckets × 8 heads = 64 bias scalars; negligible params.
conditioned on "what would happen if…"; search implemented as attention
  (64-token sequences, 6.4M params) K=4 costs ~5× latency; acceptable for
Contrastive decoding between final and an earlier pass improves selection -
*looped* transformer. LoopCD says: don't just average or take the last pass -