# Research Notes: Hyperparameter Optimization Methodology
- Parallelism: TPE suggests, ASHA prunes; Optuna supports both natively.
Stage A; trainer-side ASHA (GPU, ~1 day total):
Stage B; inference-side grid (CPU SPRT, ~1 day):
   winner net; 11 configs, but R=1/α=0 is the baseline → 10 × fast SPRT
Stage C; fine-tune chain knobs (after labeled_5m lands):
  0.9995} in stage A tail; 2 extra candidates, export both, SPRT.
  search axis; expected-positive everywhere.