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

    const NetConfig& config() const { return cfg_; }

  private:
    static float* x_buf();  // per-call token scratch (64 x d)

    NetConfig cfg_{};
    std::vector<float> w_;  // all weights, contiguous
    // offsets into w_ for each tensor group
    std::size_t piece_emb_ = 0, square_emb_ = 0, side_emb_ = 0;
    std::vector<std::size_t> layer_off_;  // per-layer base offset
    std::size_t lnP_ = 0, wfrom_ = 0, bfrom_ = 0, wto_ = 0, bto_ = 0, promoW_ = 0, promoB_ = 0;
    std::size_t lnV_ = 0, v1W_ = 0, v1B_ = 0, v2W_ = 0, v2B_ = 0;
};

}  // namespace lo::nn
