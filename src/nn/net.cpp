#include "nn/net.hpp"

#include <cmath>
#include <cstring>
#include <fstream>

namespace lo::nn {

namespace {

constexpr std::uint32_t kMagic = 0x574E4F4C;  // "LONW" LE
constexpr std::uint32_t kVersion1 = 1;
constexpr std::uint32_t kVersion2GAB = 2;
constexpr std::uint32_t kVersion3 = 3;
constexpr float kLnEps = 1e-5f;
constexpr int kMaxD = 1024;

// Square-relation bucket ids, matching trainer/model.py::gab_bucket_table:
// 0 same, 1 knight, 2 file, 3 rank, 4 diag, 5 cheb1, 6 cheb2, 7 other.
std::uint8_t gab_bucket(int s, int t) {
    const int fs = s % 8, rs = s / 8, ft = t % 8, rt = t / 8;
    const int df = std::abs(fs - ft), dr = std::abs(rs - rt);
    if (s == t) return 0;
    if ((df == 1 && dr == 2) || (df == 2 && dr == 1)) return 1;
    if (fs == ft) return 2;
    if (rs == rt) return 3;
    if (df == dr) return 4;
    const int cheb = df > dr ? df : dr;
    if (cheb == 1) return 5;
    if (cheb == 2) return 6;
    return 7;
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

// y[o] = b[o] + sum_i x[i] * W[o * in_dim + i]   (W row-major [out][in])
void linear(const float* x, const float* W, const float* b, float* y, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        float acc = b ? b[o] : 0.0f;
        const float* row = W + static_cast<std::size_t>(o) * in_dim;
        for (int i = 0; i < in_dim; ++i)
            acc += x[i] * row[i];
        y[o] = acc;
    }
}

void layernorm(const float* x, const float* g, const float* b, float* y, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float inv = 1.0f / std::sqrt(var + kLnEps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}

void softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}

// Per-call scratch (single-threaded reference path; sized for d <= kMaxD).
thread_local float q_buf[64 * kMaxD];
thread_local float k_buf[64 * kMaxD];
thread_local float v_buf[64 * kMaxD];
thread_local float ctx[64 * kMaxD];
thread_local float vec[kMaxD];
thread_local float wide[4096];
thread_local float att[64 * 64];
thread_local float hp_buf[64 * kMaxD];

}  // namespace

void Net::run_trunk(int d, int heads, int hd, float scale) const {
    const int dff = static_cast<int>(cfg_.dff);
    for (std::uint32_t l = 0; l < cfg_.layers; ++l) {
        const float* base = w_.data() + layer_off_[l];
        const float* ln1g = base;
        const float* ln1b = ln1g + d;
        const float* Wq = ln1b + d;
        const float* bq = Wq + d * d;         // bias follows its weight block
        const float* Wk = bq + d;
        const float* bk = Wk + d * d;
        const float* Wv = bk + d;
        const float* bv = Wv + d * d;
        const float* Wo = bv + d;
        const float* bo = Wo + d * d;
        const float* ln2g = bo + d;
        const float* ln2b = ln2g + d;
        const float* W1 = ln2b + d;
        const float* b1 = W1 + dff * d;
        const float* W2 = b1 + dff;
        const float* b2 = W2 + d * dff;

        // q, k, v per token
        for (int s = 0; s < 64; ++s) {
            layernorm(&x_buf()[s * d], ln1g, ln1b, vec, d);
            linear(vec, Wq, bq, &q_buf[s * d], d, d);
            linear(vec, Wk, bk, &k_buf[s * d], d, d);
            linear(vec, Wv, bv, &v_buf[s * d], d, d);
        }
        // attention: per-head dot products, softmax per head-row, and a
        // per-head value slice for the context (this is the part that makes
        // multi-head attention multi-head). GAB (blob v2) adds a learned
        // per-head bias keyed on the square-pair relation bucket.
        const bool use_gab = !gab_table_.empty();
        for (int head = 0; head < heads; ++head) {
            const float* gt = use_gab ? &gab_table_[static_cast<std::size_t>(head) * 8] : nullptr;
            for (int s = 0; s < 64; ++s) {
                const float* qp = &q_buf[s * d + head * hd];
                for (int t = 0; t < 64; ++t) {
                    const float* kp = &k_buf[t * d + head * hd];
                    float dot = 0.0f;
                    for (int i = 0; i < hd; ++i) dot += qp[i] * kp[i];
                    att[s * 64 + t] = dot * scale + (gt ? gt[gab_bucket(s, t)] : 0.0f);
                }
                softmax_inplace(&att[s * 64], 64);
                for (int i = 0; i < hd; ++i) {
                    float acc = 0.0f;
                    for (int t = 0; t < 64; ++t)
                        acc += att[s * 64 + t] * v_buf[t * d + head * hd + i];
                    ctx[s * d + head * hd + i] = acc;
                }
            }
        }
        // output projection + residual
        for (int s = 0; s < 64; ++s) {
            linear(&ctx[s * d], Wo, bo, vec, d, d);
            for (int i = 0; i < d; ++i) x_buf()[s * d + i] += vec[i];
        }
        // mlp + residual
        for (int s = 0; s < 64; ++s) {
            layernorm(&x_buf()[s * d], ln2g, ln2b, vec, d);
            linear(vec, W1, b1, wide, d, static_cast<int>(dff));
            for (int i = 0; i < static_cast<int>(dff); ++i) wide[i] = gelu(wide[i]);
            linear(wide, W2, b2, vec, static_cast<int>(dff), d);
            for (int i = 0; i < d; ++i) x_buf()[s * d + i] += vec[i];
        }
    }  // layer loop
}

void Net::policy_scores(float* out_scores) const {
    const int d = static_cast<int>(cfg_.d);
    const int dpol = static_cast<int>(cfg_.dpol);
    // Per-token policy LN into hp_buf (architecture contract: python pools
    // lnP(x), not raw x).
    for (int sq = 0; sq < 64; ++sq)
        layernorm(&x_buf()[sq * d], w_.data() + lnP_, w_.data() + lnP_ + d, &hp_buf[sq * d], d);
    // ef/et reuse the q/k scratch (d-sized per token, dpol <= d)
    for (int sq = 0; sq < 64; ++sq) {
        linear(&hp_buf[sq * d], w_.data() + wfrom_, w_.data() + bfrom_, &q_buf[sq * d], d, dpol);
        linear(&hp_buf[sq * d], w_.data() + wto_, w_.data() + bto_, &k_buf[sq * d], d, dpol);
    }
    for (int u = 0; u < 64; ++u)
        for (int v = 0; v < 64; ++v) {
            float dot = 0.0f;
            for (int i = 0; i < dpol; ++i) dot += q_buf[u * d + i] * k_buf[v * d + i];
            out_scores[u * 64 + v] = dot / std::sqrt(static_cast<float>(dpol));
        }
}

NetOutput Net::evaluate(const PositionState& pos, const NetHistory& hist) const {
    const int d = static_cast<int>(cfg_.d);
    const int heads = static_cast<int>(cfg_.heads);
    const int hd = d / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // tokens: piece code per square (1-6 white, 9-14 black, shard convention)
    for (int sq = 0; sq < 64; ++sq) {
        std::uint8_t code = 0;
        for (int t = 0; t < PIECE_NB; ++t) {
            if (pos.pieces[WHITE][t] & SQUARE_BB[sq]) code = static_cast<std::uint8_t>(t + 1);
            if (pos.pieces[BLACK][t] & SQUARE_BB[sq]) code = static_cast<std::uint8_t>(t + 9);
        }
        for (int i = 0; i < d; ++i) {
            x_buf()[static_cast<std::size_t>(sq) * d + i] =
                w_[piece_emb_ + static_cast<std::size_t>(code) * d + i] +
                w_[square_emb_ + static_cast<std::size_t>(sq) * d + i] +
                w_[side_emb_ + static_cast<std::size_t>(pos.side_to_move) * d + i];
        }
    }

    if (v3_) {
        // SPEC-BLOB-V3 inputs: castling mask, ep file, king bucket, rating 0.
        // History is not plumbed yet; its tail loads zero-init (no-op).
        const float* ce = w_.data() + castle_emb_ + static_cast<std::size_t>(pos.castling) * d;
        const int ep_idx =
            (pos.ep_square == NO_EP || pos.ep_square >= 64) ? 0 : 1 + (pos.ep_square % 8);
        const float* ee = w_.data() + ep_emb_ + static_cast<std::size_t>(ep_idx) * d;
        int ksq = 0;
        for (int sq = 0; sq < 64; ++sq)
            if (pos.pieces[pos.side_to_move][KING] & SQUARE_BB[sq]) ksq = sq;
        const int kb = (ksq / 8 / 4) * 4 + (ksq % 8 / 4);
        const float* ke = w_.data() + king_bucket_emb_ + static_cast<std::size_t>(kb) * d;
        const float* re = w_.data() + rating_emb_;  // bucket 0
        for (int sq = 0; sq < 64; ++sq)
            for (int i = 0; i < d; ++i)
                x_buf()[static_cast<std::size_t>(sq) * d + i] += ce[i] + ee[i] + ke[i] + re[i];
        // HiCo history: gated additive embeddings on from/to squares
        for (int p = 0; p < hist.n; ++p) {
            const float g = w_[hist_gate_ + p];
            const float* ef = w_.data() + hist_emb_ + static_cast<std::size_t>(p) * 2 * d;
            for (int i = 0; i < d; ++i) {
                x_buf()[static_cast<std::size_t>(hist.from[p]) * d + i] += g * ef[i];
                x_buf()[static_cast<std::size_t>(hist.to[p]) * d + i] += g * ef[d + i];
            }
        }
    }

    // Trunk pass 1 (recycling: earlier passes are weaker predictors whose
    // contrast with the final pass guides selection — LoopCD).
    run_trunk(d, heads, hd, scale);
    float scores1[64 * 64];
    const bool loopcd = recycle_ > 1 && loopcd_alpha_ != 0.0f;
    if (loopcd) policy_scores(scores1);
    for (int pass = 2; pass <= recycle_; ++pass) run_trunk(d, heads, hd, scale);

    // heads
    NetOutput out{};
    policy_scores(out.scores);
    if (loopcd) {
        for (int i = 0; i < 64 * 64; ++i)
            out.scores[i] += loopcd_alpha_ * (out.scores[i] - scores1[i]);
    }

    float pooled[kMaxD], hv[kMaxD], vhid[128];
    for (int i = 0; i < d; ++i) {
        float s = 0.0f;
        for (int sq = 0; sq < 64; ++sq) s += hp_buf[sq * d + i];
        pooled[i] = s / 64.0f;
    }
    layernorm(pooled, w_.data() + lnV_, w_.data() + lnV_ + d, hv, d);

    linear(pooled, w_.data() + promoW_, w_.data() + promoW_ + 4 * d, out.promo_logit, d, 4);

    linear(hv, w_.data() + v1W_, w_.data() + v1W_ + 128 * d, vhid, d, 128);
    for (int i = 0; i < 128; ++i) vhid[i] = gelu(vhid[i]);
    if (v3_) {
        // material bucket: non-king piece count, 8 buckets (trainer
        // material_bucket: min(7, pc*8/30))
        const int pc = static_cast<int>(popcount(occupancy_all(pos))) - 2;
        const int mb = std::min(7, std::max(0, pc * 8 / 30));
        const float* W = w_.data() + v3_v2W_ + static_cast<std::size_t>(mb) * 3 * 128;
        const float* B = w_.data() + v3_v2B_ + static_cast<std::size_t>(mb) * 3;
        for (int j = 0; j < 3; ++j) {
            float acc = B[j];
            for (int i = 0; i < 128; ++i) acc += vhid[i] * W[j * 128 + i];
            out.wdl[j] = acc;
        }
    } else {
        linear(vhid, w_.data() + v2W_, w_.data() + v2W_ + 3 * 128, out.wdl, 128, 3);
    }
    softmax_inplace(out.wdl, 3);
    return out;
}

std::optional<Net> Net::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;

    std::uint32_t magic = 0, version = 0;
    NetConfig cfg;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&cfg.d), 4);
    f.read(reinterpret_cast<char*>(&cfg.layers), 4);
    f.read(reinterpret_cast<char*>(&cfg.heads), 4);
    f.read(reinterpret_cast<char*>(&cfg.dff), 4);
    f.read(reinterpret_cast<char*>(&cfg.dpol), 4);
    if (!f || magic != kMagic) return std::nullopt;
    if (version != kVersion1 && version != kVersion2GAB && version != kVersion3)
        return std::nullopt;
    if (cfg.d > kMaxD || cfg.dff > 4096) return std::nullopt;

    auto rd_tail = [&](std::vector<float>& v, std::size_t n) {
        const std::size_t off = v.size();
        v.resize(off + n);
        f.read(reinterpret_cast<char*>(v.data() + off), static_cast<std::streamsize>(n * 4));
        return off;
    };

    Net net;
    net.cfg_ = cfg;
    const std::size_t d = cfg.d, dff = cfg.dff, dpol = cfg.dpol;

    const std::size_t per_layer = 2 * d                               // ln1
                                  + 4 * (d * d + d)                   // q k v o
                                  + 2 * d                             // ln2
                                  + dff * d + dff + d * dff + d;      // mlp
    const std::size_t total = (15 + 64 + 2) * d + per_layer * cfg.layers
                              + 2 * d                                 // lnP
                              + 2 * (dpol * d + dpol)                 // from/to
                              + 4 * d + 4                             // promo
                              + 2 * d                                 // lnV
                              + 128 * d + 128 + 3 * 128 + 3;          // value

    net.w_.resize(total);
    f.read(reinterpret_cast<char*>(net.w_.data()), static_cast<std::streamsize>(total * 4));
    if (!f) return std::nullopt;
    if (version >= kVersion2GAB) {
        // GAB table appended after the standard stream (v1 is a strict prefix)
        net.gab_table_.resize(cfg.heads * 8);
        f.read(reinterpret_cast<char*>(net.gab_table_.data()),
               static_cast<std::streamsize>(net.gab_table_.size() * 4));
        if (!f) return std::nullopt;
    }
    if (version >= kVersion3) {
        // SPEC-BLOB-V3.md tail: embeddings, HiCo tail, bucketed value head.
        // All fp32, matching trainer/model.py blob_tensors() order.
        net.v3_ = true;
        net.castle_emb_ = rd_tail(net.w_, 16 * cfg.d);
        net.ep_emb_ = rd_tail(net.w_, 9 * cfg.d);
        net.king_bucket_emb_ = rd_tail(net.w_, 16 * cfg.d);
        net.rating_emb_ = rd_tail(net.w_, 16 * cfg.d);
        net.hist_emb_ = rd_tail(net.w_, 3 * 2 * cfg.d);
        net.hist_gate_ = rd_tail(net.w_, 3);
        net.v3_v2W_ = rd_tail(net.w_, 3 * 8 * 128);
        net.v3_v2B_ = rd_tail(net.w_, 8 * 3);
        if (!f) return std::nullopt;
    }

    std::size_t off = 0;
    net.piece_emb_ = off; off += 15 * d;
    net.square_emb_ = off; off += 64 * d;
    net.side_emb_ = off; off += 2 * d;
    net.layer_off_.resize(cfg.layers);
    for (std::uint32_t l = 0; l < cfg.layers; ++l) {
        net.layer_off_[l] = off;
        off += per_layer;
    }
    net.lnP_ = off; off += 2 * d;
    net.wfrom_ = off; off += dpol * d + dpol;
    net.bfrom_ = net.wfrom_ + dpol * d;
    net.wto_ = off; off += dpol * d + dpol;
    net.bto_ = net.wto_ + dpol * d;
    net.promoW_ = off; off += 4 * d + 4;
    net.lnV_ = off; off += 2 * d;
    net.v1W_ = off; off += 128 * d + 128;
    net.v2W_ = off; off += 3 * 128 + 3;
    return net;
}


float* Net::x_buf() {
    static thread_local float buf[64 * kMaxD];
    return buf;
}

NetOutput Net::evaluate(const PositionState& pos) const {
    return evaluate(pos, NetHistory{});
}

}  // namespace lo::nn
