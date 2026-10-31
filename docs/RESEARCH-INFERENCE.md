# Research Notes: Inference-Time Techniques
## 1. LoopCD: contrastive decoding over recurrent passes
- **LoopCD-Logits:** `L = L_R + α (L_R − L_r)`; one extra output projection.
(Huginn); training-free. Gains persist at half the loop count.
  retraining; a cheap post-release strength knob.
"complex"; e.g., high pass-to-pass policy divergence) mirrors "conditional
## 3. DiffuSearch: discrete-diffusion implicit search
- Training: absorbing diffusion; mask random future tokens, predict them.
move; opponent modeling) before predicting ours. Cheap adapters (LoRA-rank-4