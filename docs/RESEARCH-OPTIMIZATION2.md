# Research Notes: Optimization Round 2: Weight Averaging, Loss Refinements, Self-Play FT
**Scope:** second-pass optimization research; techniques that improve a *trained*
## 1. Weight averaging (EMA / SWA): cheapest known win
- Verdict: **ADOPT**; implement in trainer, export both variants.
   (a) SF label or (b) game outcome; self-distillation with a quality filter.
for free. Verdict: **REJECT for now**; revisit only if EMA under-delivers.