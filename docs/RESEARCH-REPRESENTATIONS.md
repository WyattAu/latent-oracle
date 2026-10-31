# Research Notes: Representations, Values, Alternative Trunks
Our value head outputs W/D/L softmax; already a 3-bin distributional head
   output dims. Verdict: **PILOT P3**; needs the RL stage first to have a
   interpolation of eval on a bin grid; smoother than CE on a single bin.
are free (they are; the shard has everything needed):
The JEPA-lite task is the most interesting; it forces the trunk to encode
| Graph attention (pieces as nodes) | AMCTS-adjacent 2026 work; edges = attack relations | WATCH; natural fit for chess but a rewrite; revisit only if GAB under-delivers |
| CNN (Lc0-style ResNet) | proven at scale | REJECT; transformer already competitive at our size |
DiffuSearch pilot IS a lightweight dynamics model over FEN tokens; its
- Strong nets internally predict the OPPONENT'S best reply; our reply-
- Move-matching saturates while strength grows; do not over-tune on