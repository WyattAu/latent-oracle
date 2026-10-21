# Loop 9 — Openings & variety: the measurement infrastructure, audited

**Question in the loop plan:** openings.epd + opening-weight 1.5; book
diversity vs strength-measurement noise.

## What this program actually does (audited, not assumed)

- **Book**: 2000-line EPD (`openings.epd`), seeded random order, `-repeat`
  so every opening is played twice with colors swapped.
- **Statistics**: pentanomial pair model (RESEARCH-SYSTEMS §3) — pairs are
  same-opening both-colors, which is exactly fishtest's variance-reduction
  trick. Every verdict since the mid-session correction uses the engine-name
  PGN recount as authoritative, after fastchess txt mislabels were caught
  twice (MirrorAvg sign error, loop 7's int8 line).
- **Opening-weight 1.5** (training): upweights book-opening positions in the
  BC loss. No measured verdict isolates it; it predates the loop program.

## Verdicts

1. **A/B verdict integrity: sound.** Paired openings + pentanomial + name
   recount controls opening bias for all relative verdicts (this is the
   protocol that caught the d16 false positive AND the int8 mislabel).
2. **Absolute Elo is book-conditional.** All numbers in RESULTS are
   "Elo under this 2000-line book at tc=15+0.2" — not field Elo. Recorded as
   an interpretation rule.
3. **Book diversity ceiling** (measured in loop 6): 2000 seeds cap self-play
   diversity; for SPRT A/B use, 2000 paired lines at 300-900 games is
   adequate (SE floor from openings is below the game-noise floor).

**Spec rule going forward:** any claim of absolute strength vs the outside
world requires either a bigger/differentiated book (fishtest-style) or a
gauntlet vs external references. Internal A/B verdicts keep the current book.

**Loop 9 verdict:** infrastructure audited and validated; interpretation
rule recorded; no change needed for internal verdicts.
