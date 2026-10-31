# Research Notes: External Resources: NNUE Corpus, KataGo, Lc0 Production, Maia
**Scope:** non-arXiv sources; the chess-engine engineering literature
### 1.1 Re-validations (we already do these: independent confirmation)
   at export. Fixes slow learning for low-coverage buckets; directly
   transformer uses GELU/SwiGLU; different family; but SCReLU is a
   marlinflow, Grapheus; their practices (SLD learning-rate ramp, union
## 2. KataGo: "Accelerating Self-Play Learning in Go" (arXiv 1902.10565)
| **Playout cap randomization**; train on positions from cheap searches, sample expensive searches occasionally | Mixed-depth labeling: 20% of positions at d16, 80% at d10, weight loss by depth; cuts labeling cost ~2-3× with minimal quality loss. **ADOPT for the next labeling round** (the Lc0 3M shard), keep uniform d14 for the current 5M. |
| **Auxiliary targets** (ownership map, score, opponent next-move) | Our JEPA-lite + reply-prediction + eval-delta aux plan (RESEARCH-REPRESENTATIONS §2); same principle, validated in Go. **Re-validated.** |
| **Global pooling heads** | Our value head mean-pools; same structure. Validated. |
- **EngineBattle** (Jul 2025): streaming/testing tool for Lc0; functionally
  specialization became Lc0's public showcase; a plausible post-v1.0
## 4. Maia (maiachess.com): rating-conditioned human modeling
     showed per-style experts beat blended; conditioning is the cheap
- Maia FAQ page is JS-rendered (no static content); the arXiv paper