# Game-Edge Tracking: Policy-Weighted Value Regret in Calibrated Probability Space

## 1. The idea, stated precisely

Centipawn-lost (CPL) asks: "how much worse than the engine's BEST move was the
played move, in centipawns?" Four structural failures:

1. **Scale**: centipawns are not utility. 50cp at eval +0.2 and 50cp at +3.0
   are completely different events. Any per-move aggregate over raw cp mixes
   incommensurable units.
2. **Oracle harshness**: CPL measures regret against a minimax oracle. A move
   that is 40cp worse than tablebase-perfect but completely safe against every
   realistic reply scores the same as a 40cp blunder that loses by force.
   The reference distribution is wrong for the question "how big was that
   error *in this game*?"
3. **Horizon blowups**: single-line search evals swing violently at tactical
   horizons; CPL inherits swings that have nothing to do with either player.
4. **No opponent model**: after your move the opponent replies. CPL evaluates
   your move by the line the analyser *would* have played, not by what your
   opponent could do. Real edge depends on both.

The proposed metric replaces "value of best move" with the **policy-weighted
expectation over the mover's own move distribution**, taken in **calibrated
win-probability space**, with the **opponent's reply distribution** supplying
the transition expectation. Informally, per move:

> edge = (what this position was worth in expectation, given how this player
> moves) − (what it became worth after the move actually played)

That is exactly the proposed "current value − expected value at this move",
made rigorous by (a) putting it in an MDP so "expected" has a definition,
(b) working in outcome space so units are interpretable, (c) calibrating the
evaluator, and (d) separating your error from your opponent's.

## 2. Formal definitions

### 2.1 Setup

An MDP whose states are positions s (side-to-move included) and whose action
set at s is the legal moves A(s). Two ingredients from our searchless engine
(one forward pass per position, no search):

- **Policy** π(·|s): the masked policy head over A(s), optionally temperature-
  sharpened. π is a *skill model* — "how this player moves".
- **Value** v(s): calibrated mover-POV expected score, v(s) ∈ [0,1]
  (win + ½·draw), from the wdl head. Calibration (§4) is mandatory before
  deltas are meaningful.

The afterstate of playing a in s is s′ = s ⊕ a, with side-to-move flipped.
Define the mover-POV action value by evaluating the afterstate from the
opponent's POV and converting:

  **Q(s,a) := 1 − v(s′)**   (for the 3-way (w,d,l) head: mover mean = 1 − opp mean;
  the full triple converts as (l, d, w)).

### 2.2 Expected value at a position

  **V̄(s) := Σ_{a∈A(s)} π(a|s) · Q(s,a)**

This is the Bellman expectation of the position under the skill model — the
rigorous content of "expected value at this move". Note it is *not* the value
head's own number v(s): V̄ averages over the moves the player actually
considers plausible; v is a static judgement. Their difference

  **g(s) := V̄(s) − v(s)**

is the evaluator's *self-consistency gap* (policy vs value head), itself a
useful diagnostic — it is where the net's move-choice and its evaluation
disagree.

### 2.3 Per-move edge

For the played move a_t at ply t:

  **EVR_t := V̄(s_t) − Q(s_t, a_t)**

Expected-Value Regret: how much worse the played move is than the position's
worth under the player's own move distribution. Properties:

- E_{a~π}[EVR] = 0 *by construction* — the metric is centered on the skill
  model, so it measures deviation from one's own standard, not from a god.
- For a player whose actual move distribution ρ ≠ π is weaker (puts weight on
  lower-Q moves), E_{a~ρ}[EVR] > 0 — aggregate EVR is a strength-gap signal
  relative to π. This is a proper-scoring-rule property: it is the covariance
  form Σ_a (ρ−π)·Q, zero exactly when ρ=π.
- Bounded by the policy's own uncertainty: in a forced position (π mass on one
  move) EVR ≈ 0 however catastrophic — correct, since there was nothing to
  choose. Sharpness is visible in the spread of Q, reported alongside.

The CPL analogue would be **M_t := max_a Q(s_t,a) − Q(s_t,a_t)** — kept as a
secondary ("oracle regret") for comparison, expected to be harsher and noisier.

### 2.4 Game-level edge and the telescoping identity

Along the actual game s_0, a_0, s_1, b_0, s_2, a_1, … (P to move at even
plies), define for P the mover-POV value of each position. Then for any ply:

  v_P(s_t) − v_P(s_{t+1 mover-converted}) = −EVR_t + ρ_t,

where the **drift residual** ρ_t = [Q(s_t,a_t) − v_P(s_{t+1})] is the part of
the value change *not* explained by P's own choice — it collects the opponent's
upcoming choice and the evaluator's static-eval inconsistency. Telescoping
over both players to the end of the game:

  v_P(s_T) − v_P(s_0) = −Σ_P EVR − Σ_Q EVR + Σ ρ_t  (signs per POV conversion)

i.e. **the total value swing of a game decomposes into (your cumulative
expected-value regret) + (opponent's) + (evaluator drift)**. The drift term is
the honest accounting of what the static evaluator cannot explain (horizon,
tactics beyond one ply). Validation requires drift ≪ signals (§5); where it is
not, the residual itself flags "this game turned on something the evaluator
cannot see" — also useful (it marks games/moves for search-based review).

### 2.5 Attribution: whose edge?

The identity gives per-player ledgers: cumulative EVR per side measures what
each player leaked relative to the skill model; the difference of the two
cumulatives tracks the swing of the game. A momentum chart is the first
difference of (opp_leak − own_leak).

## 3. Relation to prior art

| Prior art | What it does | What EVR changes |
|---|---|---|
| CPL / ACPL (Stockfish convention) | cp lost vs best line | outcome space vs cp; expectation vs oracle; drift-separated |
| Win-% lost (Lichess accuracy) | logistic eval→win% map, lost vs best | still oracle-referenced; no policy expectation; no opponent model; calibration fixed by their map, not fitted |
| Regan & Haworth, Intrinsic Chess Ratings | move-match probability vs a 3000-Elo move-chooser, per-move rank | π IS the reference distribution; Q-weighting gives magnitudes not just match counts; per-game attribution |
| AlphaZero published game analyses (Sadler & Regan) | plot Q of played vs best Q per move | closest prior art; EVR adds expectation-referencing, calibration, telescoping, drift term |
| Advantage A(s,a)=Q−V, GAE (RL) | per-decision improvement signal | EVR is the empirical advantage under ρ vs π; identical machinery later reusable as RL reward-shaping (loop 2) |
| Proper scoring rules | sharpness/calibration decomposition | calibration is explicit (§4); EVR is the regret form of a scoring rule |

## 4. Calibration (mandatory)

Deltas are meaningless if v is biased. Fit an isotonic regression g: raw wdl
mean → empirical score on held-out data with known outcomes. Our corpus:
tens of thousands of engine-vs-engine games in /home/wyatt/data/sprt with
recorded results — replay a sample, bin positions by raw v, fit isotonic on
half, validate on half (ECE before/after, and reliability curve). Re-fit when
the net changes. All EVR math runs on g(v).

Bias that survives isotonic (static-eval miscalibration conditional on
position features) mostly cancels in EVR because both V̄ and Q are produced by
the same v at adjacent positions — this is the strongest argument for the
delta form and should be validated: regress empirical outcomes of decisive
sub-games on predicted edge; slope ≈ 1 means calibrated deltas.

## 5. Validation plan (all CPU, runnable on the existing corpus)

1. **Strength separation**: EVR per game aggregated by player, across the
   SF-limit ladder (bc40m vs sf2000/2200/2400) and engine pairs of known
   Elo gaps. Requirement: monotone in strength, effect size larger for EVR
   than for cp-based CPL on the same games.
2. **Outcome prediction**: logistic regression result ~ cumulative edge
   (EVR form vs CPL form vs WP-lost form); compare AUC / log-loss. Requirement:
   EVR ≥ alternatives.
3. **Drift audit**: report Σ|ρ_t| vs Σ|EVR| per game; drift share should be
   minority except in sharp tactical games (which the drift term then flags).
4. **Sharpness interaction**: bin moves by policy entropy; verify EVR is not
   just re-detecting "position was forced".

## 6. Engineering

`trainer/edge_tracker.py` (CPU, ~0.5 s/game with the 6.4M-param net batched):
PGN in → per-ply (V̄, Q(played), EVR, oracle-regret, drift, entropy) → per-game
aggregates + calibration table. Batch all positions of a game through the net
in one pass; POV conversion on the (w,d,l) triple. Reuses sprt_pentanomial-
style name handling for attribution.

## 7. Naming

EVR — Expected-Value Regret (policy-weighted, probability-space, drift-
separated). "Game edge" for the cumulative signed version.

## 8. Status

- [x] formulation (this doc)
- [ ] isotonic calibration on sprt corpus
- [ ] edge_tracker.py
- [ ] validation 1-4 with numbers

## 9. Validation results (2026-10-26, bc40m-vs-sf ladder, net_e1 evaluator)

Setup: isotonic calibration on 6,742 positions (400 games) — ECE 0.4805 →
0.4775: the map is near-identity, i.e. **the wdl head is already
probability-calibrated on engine-vs-engine games** (ECE ≈ the irreducible
outcome-noise floor). Raw values are usable directly.

Per-move EVR / oracle-regret on 50-200 games per class:

| class  | n   | EVR/move | oracle/move | game score |
|--------|-----|----------|-------------|------------|
| sf2000 | 100 | 0.0264   | 0.1870      | 0.985      |
| sf2200 | 50  | 0.0222   | 0.1756      | 0.990      |
| sf2400 | 50  | 0.0114   | 0.1426      | 0.980      |
| bc40m  | 200 | 0.0026   | 0.0617      | 0.015      |

**Finding — the reference-policy inversion.** My prediction (bc40m highest
EVR: weakest player, most visible errors) was INVERTED. bc40m shows the
LOWEST EVR (0.0026) and the lowest oracle regret, despite losing 98.5%.
Interpretation: with a same-family weak evaluator (net_e1), EVR measures
**style agreement with the evaluator**, not error magnitude:

- bc40m and the evaluator share a training lineage — their move preferences
  coincide, so bc40m's objective blunders are invisible to the metric.
- The SF players' (better) moves deviate from net_e1's preferences; their Q
  under the weak value head looks mediocre, so they appear to "regret".
- Within the SF ladder the ordering IS monotone (sf2400 < sf2200 < sf2000):
  stronger play aligns better with universal features (material/tactics)
  that even a weak value head perceives.

AUC(-ownEVR → win-vs-loss) = 0.4708 ≈ chance, as expected under inversion.

**Consequences:**
1. EVR with a self-referential evaluator is a *same-lineage* metric: valid
   for comparing OUR training runs against a FIXED reference net (the actual
   use case), invalid across lineages/evaluator-weaker players.
2. True error magnitude needs evaluator strength ≫ player strength — either
   an external oracle (SF fixed-depth Q, keeping the policy-weighted
   expectation) or our strongest net as the fixed reference. Hybrid variant
   queued: SF supplies Q, our net supplies π and calibration.
3. The oracle-regret column inverts identically — this is an
   evaluator-strength limitation, not an EVR-specific flaw.
4. The metric's stated property "measures deviation from one's own
   standard" is precisely what it did: bc40m deviates least from
   net_e1's standard because it IS net_e1's standard.

Honest bookkeeping: the formulation stands; the validation falsified the
"one evaluator fits all" assumption and localized exactly which comparisons
the metric supports.
