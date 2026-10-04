# Research Notes — External Resources: NNUE Corpus, KataGo, Lc0 Production, Maia

**Scope:** non-arXiv sources — the chess-engine engineering literature
(Chess Programming Wiki), Go efficiency research (KataGo), Lc0's production
blog, and human-modeling work (Maia). These encode decades of what actually
works in production chess engines.

---

## 1. NNUE engineering corpus (Chess Programming Wiki, "NNUE" + "Stockfish NNUE")

The most battle-tested neural-eval knowledge base in chess. Key items and
their transfer to latent-oracle:

### 1.1 Re-validations (we already do these — independent confirmation)
| Our choice | NNUE equivalent | Status |
|---|---|---|
| File-mirror augmentation (`--mirror`) | Horizontal mirroring ("1–2× more effective data") | **validated** |
| Value target = interpolation of game result + sigmoid(eval/scale) | NNUE target: `result + sigmoid(eval/SCALE)` blend | **validated** (same shape) |
| INT8 quantization with explicit dequant scales | QA/QB integer scaling scheme | validated approach |
| Game-phase data balancing (RESEARCH-DATA §2.2) | Output buckets by non-king piece count; LayerStacks | **validated** |

### 1.2 New adoptable ideas
1. **Material-conditioned output buckets** (blob v3 candidate): split the
   value head's output weights into 8 buckets indexed by piece count.
   Cheap (8× head params), proven in every top engine. Alternative: keep
   one head and add a material one-hot to the value input (smaller).
2. **Feature factorizer training**: when bucketing, add a always-active
   "shadow" bucket during training, then fold its weights into each bucket
   at export. Fixes slow learning for low-coverage buckets — directly
   relevant to our endgame under-representation.
3. **King input buckets**: several accumulator/head weight sets chosen by
   king region (MoE-lite). For us: king-square bucket embedding added to
   token 0 (or all tokens). Candidate for blob v3 alongside castling/ep.
4. **SCReLU activation** (clamp(x,0,1)² at the output layer): "most
   prevalent and produces the strongest network" for NNUE eval nets. Our
   transformer uses GELU/SwiGLU — different family — but SCReLU is a
   candidate for the VALUE HEAD's hidden activation at the next blob bump.
5. **Trainer ecosystem references**: Bullet (Rust), nnue-pytorch,
   marlinflow, Grapheus — their practices (SLD learning-rate ramp, union
   data filtering, massive batching, GPU-resident dataloaders) are worth
   mining when we scale training throughput.

### 1.3 Not transferable
- Efficiently-updatable accumulator: requires incremental make-move
  structure; our transformer recomputes per position anyway (no search
  tree → few evaluations per move → no need).
- Perspective concatenation (stm/nstm): we inject side-to-move as an
  embedding; equivalent role.

---

## 2. KataGo — "Accelerating Self-Play Learning in Go" (arXiv 1902.10565)

50× compute reduction over AlphaZero-style baselines via:

| KataGo technique | Transfer to us |
|---|---|
| **Playout cap randomization** — train on positions from cheap searches, sample expensive searches occasionally | Mixed-depth labeling: 20% of positions at d16, 80% at d10, weight loss by depth — cuts labeling cost ~2-3× with minimal quality loss. **ADOPT for the next labeling round** (the Lc0 3M shard), keep uniform d14 for the current 5M. |
| **Auxiliary targets** (ownership map, score, opponent next-move) | Our JEPA-lite + reply-prediction + eval-delta aux plan (RESEARCH-REPRESENTATIONS §2) — same principle, validated in Go. **Re-validated.** |
| **Game-phase-based training selection** | Already noted (RESEARCH-DATA §2.2). |
| **Global pooling heads** | Our value head mean-pools — same structure. Validated. |

---

## 3. Lc0 production blog (lczero.org/blog)

- **"Fine-Tuning Lc0 Network for Odds Games"** (Nov 2024): fine-tuning the
  strong base net on scenario-specific games (contempt parameter +
  opponent-emulation data) produced a knight-odds net that scored
  **+14 −2 −2 vs GM Lenderman** at 15'+10". Validates the warm-start +
  scenario-fine-tune pattern that structures our entire chain (base → AV →
  quality → scenario). Contempt parameter is a product feature to note for
  persona/odds extensions later.
- **EngineBattle** (Jul 2025): streaming/testing tool for Lc0 — functionally
  parallels our fastchess + SPRT + keep-best infrastructure; nothing to
  adopt, but confirms our evaluation stack is standard-issue.
- Odds-bot program overall: after strength milestones, scenario
  specialization became Lc0's public showcase — a plausible post-v1.0
  product axis for us (persona nets, odds nets) built on the same
  fine-tuning machinery.

---

## 4. Maia (maiachess.com) — rating-conditioned human modeling

- Maia trains **9 separate nets** (1100 → 1900) on rating-bucketed human
  games, all sharing the board-transformer architecture.
- Two adoptable ideas:
  1. **Rating conditioning**: add a rating-bucket embedding to the input
     (one net covering 2000-2600 instead of a single blended policy).
     Cheap (1 embedding row), improves data efficiency across the rating
     range, and enables "play like ~2400" persona features. **BLOB V3
     CANDIDATE.** Caveat: Mixture-of-Masters (RESEARCH-ARCHITECTURES §2)
     showed per-style experts beat blended — conditioning is the cheap
     middle ground.
  2. **Style/persona products**: our Lichess data retains player ratings;
     fine-tuned persona nets are a v1.x product axis adjacent to Lc0's
     odds bots.
- Maia FAQ page is JS-rendered (no static content) — the arXiv paper
  ("Aligning Superhuman AI with Human Behavior: Chess as a Model System",
  2020) is the citable source.

---

## 5. Book: "Neural Networks for Chess" (Dominik Klein, 2021)

Pedagogical coverage of AlphaZero-style, Maia-style, and NNUE nets with
code. Reference for provenance-friendly reimplementation checks; no new
techniques beyond the sources above.

---

## 6. Actionable summary (new items only)

| Item | Where | Priority |
|---|---|---|
| Mixed-depth labeling (KataGo playout-cap analog) | next labeling round (Lc0 shard) | P2 |
| Material output buckets | blob v3 | P2 (with castling/ep inputs) |
| Feature factorizer for bucket training | blob v3 trainer | P2 |
| King-square bucket embedding | blob v3 | P3 |
| SCReLU value-head activation | blob v3 | P3 |
| Rating-conditioning embedding | blob v3 or persona track | P3 |
| Contempt / odds persona nets | product track | v1.x |
