# Research Notes — Round 9: Robustness Mining, Schedule Science, Opening Weighting

**Scope:** three cheap, orthogonal additions (A1–A3) plus two watch items.
Everything slots into the AV/RL phases already planned; nothing changes the
critical path.

---

## A1. Blind-spot mining + off-distribution eval gate (ADOPT)

**Evidence:** "Adversarial Policies Beat Superhuman Go AIs" (Wang et al.,
2022): adversaries trained against KataGo win >97% by exploiting positions
rare in the training distribution — and the vulnerability *survives
adversarial retraining*. Lesson for us: human-game data leaves whole regions
of position space unvisited, and our net will have invisible failure modes
there.

### Two concrete actions
1. **Surprise mining (focused replay):** over our own archived SPRT/self-play
   PGNs, score every position with the net's value head; surprise =
   |predicted pwin − realized game outcome| (attributed at game end).
   High-surprise positions (top 2%) become high-weight (3×) training data in
   the next fine-tune. This is the chess instance of focused replay —
   targeted at the net's OWN failures, not a fixed heuristic.
   - Implementation: `mine_blindspots.py` (PGN → positions + outcomes →
     value-head pass → sidecar like the TB labels). Reuses the
     `--tb-sidecar` consumption machinery with a different generator.
2. **Off-distribution eval leg:** openings.epd is human-openings only. Add a
   **random-opening SPRT leg** (500 random legal 6-12 ply starts) as a
   standing robustness gate. A net that gains Elo on human openings but
   collapses on random starts has learned human priors, not chess — and
   would be exploitable exactly like KataGo was.

### Verdict
Mining: ADOPT at the first post-AV fine-tune. Random-openings gate: ADOPT
into the eval suite now (one script + one fastchess leg).

---

## A2. Warmup-Stable-Decay schedule (ADOPT for the d10 pretrain)

WSD (constant LR, short decay tail) matches or beats cosine at fixed compute
for multi-epoch pretraining and allows stopping early without losing the
decay benefit (widely adopted in LLM pretraining since 2023; e.g., MiniMax-01
and successors).

- Applies to the **d10 4M AV pretrain** (the long stage): 90% of steps at
  stable LR, linear decay to 10% over the final 10%.
- The d16 fine-tune and other short stages keep constant LR (they're already
  effectively "stable-only").
- Implementation: `--sched wsd` in train.py (~15 lines; cosine remains
  default for legacy runs).

### Verdict: ADOPT (implemented this round).

---

## A3. Opening-phase loss upweighting (PILOT)

Openings are knowledge-thick and position-only nets must re-derive them from
piece-layout priors. Upweight fullmove ≤ 12 records by 1.5–2× in the AV
phase to force opening-theory consolidation. Cheap (a weight term on
`fullmove`), risk: overfits book lines (mirror aug + quality filter
mitigate). The fullmove value already rides in every shard record.

Also noted: book-lite (top-N shard openings, RESEARCH-DATA §5) stays a
separate product feature — do not blur book and net.

### Verdict: PILOT (flag `--opening-weight`).

---

## A4. Layer-wise LR decay — WATCH
LLM practice (0.75–0.95 depth decay) targets 30-100 layer stability; at 8
layers the effect should be noise. Revisit only if deep-vs-shallow ablations
happen (d320/6L alternative in RESEARCH-REPRESENTATIONS §5).

## A5. Gradient-noise-scale batch audit — WATCH
One-time GNS measurement would tell us if batch 512 is far below optimal
(and whether gradient accumulation buys speed on the d10 run). 6GB memory
caps real batch growth; WATCH until the GPU frees.

---

## Implementation status this round

| Item | Status |
|---|---|
| A2 WSD schedule | **implemented** (`--sched wsd`) |
| A1 random-openings gate | **implemented** (`random_openings.py` + eval-suite doc note) |
| A1 surprise mining | script pending first self-play archive (uses --tb-sidecar consumption, ready) |
| A3 opening upweight | **implemented** (`--opening-weight`) |
