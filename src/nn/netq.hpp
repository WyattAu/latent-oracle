#pragma once

// INT8 inference for ChessNet ("LOQW" v1 blob — trainer/export_int8.py is the
// definitive layout). Quantized linears use the exact centered-madd integer
// kernel (per-tensor u8 activations with zero point, s8 weights, s32
// accumulate; the -128 centering keeps every s16 partial product in range, so
// the integer path has NO saturation error). Wo, promo, V2, embeddings,
// LayerNorms, GELU and softmax stay FP32.
//
// Policy-scale note: Wfrom's output is not pre-divided by sqrt(dpol) in the
// integer path; the division happens on the final dot. Both sides of the
// parity contract (this file and the Python quantized simulation) do this
// identically.

#include "position.hpp"
#include <vector>

#include <cstdint>
#include <optional>
#include <string>

namespace lo::nn {

struct NetQConfig {
    std::uint32_t d = 0;
    std::uint32_t layers = 0;
    std::uint32_t heads = 0;
    std::uint32_t dff = 0;
    std::uint32_t dpol = 0;
};

struct NetQOutput {
    float wdl[3];
    float scores[64 * 64];
    float promo_logit[4];
};

class NetQ {
  public:
    static std::optional<NetQ> load(const std::string& path);

    NetQOutput evaluate(const PositionState& pos) const;

  private:
    struct QL {
        std::size_t w_off = 0;   // iw_: s8 weights, row-major [out][in]
        std::size_t b_off = 0;   // f_: fp32 bias
        std::size_t rs_off = 0;  // rs_: s32 per-row weight sums
        std::size_t ws_off = 0;  // f_: fp32 weight scale
        float act_s = 1.0f;
        std::int32_t act_zp = 0;
        int in_dim = 0;
        int out_dim = 0;
    };

    // quantized linear: dequant(dot(quantize(in), w)) + b — exact integer dot
    void qlinear(const float* in, const QL& q, float* out) const;

    NetQConfig cfg_{};
    std::vector<float> f_;         // fp32 tensors
    std::vector<signed char> iw_;  // int8 weights
    std::vector<std::int32_t> rs_; // row sums
    std::vector<QL> ql_;           // 5 per layer (Wq Wk Wv W1 W2) + Wfrom Wto V1

    std::size_t emb_off_ = 0, sq_off_ = 0, side_off_ = 0;
    std::size_t ln1_off_[32] = {}, wo_off_[32] = {}, ln2_off_[32] = {};
    std::size_t lnP_ = 0, promo_ = 0, lnV_ = 0, v2_ = 0;
    std::size_t ql_from_ = 0;  // ql_ index of layer 0's Wq

    static float* x_buf();
};

}  // namespace lo::nn
