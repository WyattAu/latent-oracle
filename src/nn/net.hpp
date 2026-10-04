#pragma once

// Reference FP32 inference for ChessNet (see latent-oracle-data/trainer/
// model.py for the architecture contract). This is the M1 measurement path:
// exact, simple, single-threaded. INT8 VNNI kernels replace only the matmul
// internals in M2, behind this same interface.
//
// Weight blob layout ("LONW", version 1): u32 magic, u32 version, u32 d,
// layers, heads, dff, dpol, then f32 arrays row-major in the order written by
// trainer/model.py::blob_tensors().

#include "position.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lo::nn {

struct NetConfig {
    std::uint32_t d = 0;
    std::uint32_t layers = 0;
    std::uint32_t heads = 0;
    std::uint32_t dff = 0;
    std::uint32_t dpol = 0;
};

struct NetOutput {
    float wdl[3];                        // softmaxed, side-to-move POV
    float scores[64 * 64];               // scores[from * 64 + to]
    float promo_logit[4];                // pooled promotion logits (N B R Q)
};

class Net {
  public:
    static std::optional<Net> load(const std::string& path);

    NetOutput evaluate(const PositionState& pos) const;

    // Inference-time recycling (looped trunk) + LoopCD contrastive decoding:
    // pass 1 scores S1 are kept, the trunk runs `recycle` times total, and the
    // final scores become S_R + loopcd_alpha * (S_R - S1). recycle = 1 (the
    // default) reproduces the plain single pass bit-for-bit. Training-free
    // strength knob (LoopCD, arXiv 2610.02185).
    void set_inference(int recycle, float loopcd_alpha) {
        recycle_ = recycle < 1 ? 1 : recycle;
        loopcd_alpha_ = loopcd_alpha;
    }

    const NetConfig& config() const { return cfg_; }

  private:
    static float* x_buf();  // per-call token scratch (64 x d)

    void run_trunk(int d, int heads, int hd, float scale) const;
    void policy_scores(float* out_scores) const;  // uses hp_buf scratch

    NetConfig cfg_{};
    std::vector<float> w_;  // all weights, contiguous
    // offsets into w_ for each tensor group
    std::size_t piece_emb_ = 0, square_emb_ = 0, side_emb_ = 0;
    std::vector<std::size_t> layer_off_;  // per-layer base offset
    std::size_t lnP_ = 0, wfrom_ = 0, bfrom_ = 0, wto_ = 0, bto_ = 0, promoW_ = 0, promoB_ = 0;
    std::size_t lnV_ = 0, v1W_ = 0, v1B_ = 0, v2W_ = 0, v2B_ = 0;
    // blob v2 (GAB): learned per-head bias over square-relation buckets
    std::vector<float> gab_table_;  // heads * 8; empty when absent
    // blob v3 (SPEC-BLOB-V3.md): castle/ep/king/rating embeddings, HiCo
    // history tail, material-bucketed value head. The engine supplies
    // castle/ep/king from the position; rating defaults to bucket 0 and
    // history is not yet plumbed (zero-init tail makes both no-ops).
    std::size_t castle_emb_ = 0, ep_emb_ = 0, king_bucket_emb_ = 0, rating_emb_ = 0;
    std::size_t hist_emb_ = 0, hist_gate_ = 0;  // loaded, unused until plumbed
    std::size_t v3_v2W_ = 0, v3_v2B_ = 0;       // bucketed value head (24x128 + 24)
    bool v3_ = false;
    int recycle_ = 1;
    float loopcd_alpha_ = 0.0f;
};

}  // namespace lo::nn
