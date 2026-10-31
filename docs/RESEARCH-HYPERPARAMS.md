# Hyperparameter Recommendations: latent-oracle 6.4M
| Muon LR |; | 0.02, momentum 0.95, Nesterov | standard small-model range |
| Precision | fp16 AMP | keep (bf16 if GPU allows; 2060 is fp16-only) |; |
| Epochs | 3 | keep (see data ratio §4) |; |
| **Keep-best gate** |; |; |; | SPRT each export vs current best; only promote winners (gold-standard study §1.1.5) |