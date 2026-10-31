# Research Notes: Training
Agent Strong"; 100+ controlled runs against a fixed rule-based expert.
1. **Trust-region-style updates**; stable policy improvement.
2. **Well-aimed reward**; reward pointed at the actual objective.
3. **Curriculum of tougher opponents**; opponents graduated in strength.
4. **Warm starting**; initialize from a prior trained model.
5. **Keeping the best checkpoint**; evaluate-and-retain, never regress.
- **Imitation learning / DAgger**; unhelpful at lightweight scale. *This is a
- **Extra capacity**; "extra capacity does little to break the ceiling." The
approximation) before applying the update; the update direction is
matches the logistic k=0.00368208 already used in our value loss (1/k ≈ 272 -
difficulty ordering**; proxy = |eval| (decisiveness) or game phase.
1. Decisiveness weighting (implemented); implicit, no data reordering.