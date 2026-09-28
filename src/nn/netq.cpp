#include "nn/netq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace lo::nn {

namespace {

constexpr std::uint32_t kMagic = 0x57514F4C;  // "LOQW" LE
constexpr std::uint32_t kVersion = 1;
constexpr float kLnEps = 1e-5f;
constexpr int kMaxD = 1024;
constexpr int kMaxDff = 4096;
constexpr int kMaxLayers = 32;

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
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

void linear_f32(const float* x, const float* W, const float* b, float* y, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        float acc = b[o];
        const float* row = W + static_cast<std::size_t>(o) * in_dim;
        for (int i = 0; i < in_dim; ++i) acc += x[i] * row[i];
        y[o] = acc;
    }
}

thread_local float q_x[64 * kMaxD];
thread_local float q_h[64 * kMaxD];
thread_local float q_q[64 * kMaxD];
thread_local float q_k[64 * kMaxD];
thread_local float q_v[64 * kMaxD];
thread_local float q_att[64 * 64];
thread_local float q_vec[kMaxD];
thread_local float q_wide[kMaxDff];
thread_local float q_pooled[kMaxD];
thread_local float q_hv[kMaxD];
thread_local float q_vhid[128];
thread_local std::uint8_t q_a[kMaxD];

}  // namespace

// The exact integer kernel: out[o] = dequant(dot(a_centered, w_row)) + b[o].
// AVX2 widens to s16 and subtracts the 128 center before madd — partial
// products stay <= 2*128*127 = 32512, so the s16 madd cannot saturate and the
// AVX2 and scalar paths produce identical integers.
void NetQ::qlinear(const float* in, const QL& q, float* out) const {
    const int in_dim = q.in_dim, out_dim = q.out_dim;
    const signed char* w = iw_.data() + q.w_off;
    const std::int32_t* rs = rs_.data() + q.rs_off;

    for (int i = 0; i < in_dim; ++i) {
        const long v = std::lround(in[i] / q.act_s) + q.act_zp;
        q_a[i] = static_cast<std::uint8_t>(std::clamp(v, 0L, 255L));
    }

#if defined(__x86_64__)
    static const bool has_avx2 = __builtin_cpu_supports("avx2") != 0;
    if (has_avx2) {
        const __m256i v128 = _mm256_set1_epi16(128);
        for (int o = 0; o < out_dim; ++o) {
            const signed char* wr = w + static_cast<std::size_t>(o) * in_dim;
            __m256i acc = _mm256_setzero_si256();
            int i = 0;
            for (; i + 16 <= in_dim; i += 16) {
                const __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q_a + i));
                const __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + i));
                const __m256i a16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(a8), v128);
                const __m256i w16 = _mm256_cvtepi8_epi16(w8);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, w16));
            }
            std::int32_t tail = 0;
            for (; i < in_dim; ++i) tail += (int(q_a[i]) - 128) * int(wr[i]);
            std::int32_t acc_arr[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc_arr), acc);
            std::int32_t total = tail;
            for (int lane = 0; lane < 8; ++lane) total += acc_arr[lane];
            out[o] = (static_cast<float>(total + (128 - q.act_zp) * rs[q.rs_off + o]) *
                      q.act_s * f_[q.ws_off + o]) +
                     f_[q.b_off + o];
        }
        return;
    }
#endif
    for (int o = 0; o < out_dim; ++o) {
        const signed char* wr = w + static_cast<std::size_t>(o) * in_dim;
        std::int32_t total = 0;
        for (int i = 0; i < in_dim; ++i) total += (int(q_a[i]) - 128) * int(wr[i]);
        out[o] = (static_cast<float>(total + (128 - q.act_zp) * rs[q.rs_off + o]) * q.act_s *
                  f_[q.ws_off + o]) +
                 f_[q.b_off + o];
    }
}

std::optional<NetQ> NetQ::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;

    std::uint32_t magic = 0, version = 0;
    NetQConfig cfg;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&cfg.d), 4);
    f.read(reinterpret_cast<char*>(&cfg.layers), 4);
    f.read(reinterpret_cast<char*>(&cfg.heads), 4);
    f.read(reinterpret_cast<char*>(&cfg.dff), 4);
    f.read(reinterpret_cast<char*>(&cfg.dpol), 4);
    if (!f || magic != kMagic || version != kVersion) return std::nullopt;
    if (cfg.d > kMaxD || cfg.layers > kMaxLayers || cfg.dff > kMaxDff) return std::nullopt;

    NetQ net;
    net.cfg_ = cfg;
    const std::size_t d = cfg.d, dff = cfg.dff, dpol = cfg.dpol;

    auto rd_f32 = [&](std::size_t n) {
        const std::size_t off = net.f_.size();
        net.f_.resize(off + n);
        f.read(reinterpret_cast<char*>(net.f_.data() + off), static_cast<std::streamsize>(n * 4));
        return off;
    };
    auto rd_s8 = [&](std::size_t n) {
        const std::size_t off = net.iw_.size();
        net.iw_.resize(off + n);
        f.read(reinterpret_cast<char*>(net.iw_.data() + off), static_cast<std::streamsize>(n));
        return off;
    };
    auto rd_s32 = [&](std::size_t n) {
        const std::size_t off = net.rs_.size();
        net.rs_.resize(off + n);
        f.read(reinterpret_cast<char*>(net.rs_.data() + off), static_cast<std::streamsize>(n * 4));
        return off;
    };
    auto rd_ql = [&](int in_dim, int out_dim) {
        QL q{};
        q.in_dim = in_dim;
        q.out_dim = out_dim;
        f.read(reinterpret_cast<char*>(&q.act_s), 4);
        f.read(reinterpret_cast<char*>(&q.act_zp), 1);
        q.w_off = rd_s8(static_cast<std::size_t>(in_dim) * out_dim);
        q.ws_off = rd_f32(1);
        q.b_off = rd_f32(out_dim);
        q.rs_off = rd_s32(out_dim);
        net.ql_.push_back(q);
    };

    net.emb_off_ = rd_f32(15 * d);
    net.sq_off_ = rd_f32(64 * d);
    net.side_off_ = rd_f32(2 * d);

    net.ql_from_ = net.ql_.size();
    for (std::uint32_t l = 0; l < cfg.layers; ++l) {
        net.ln1_off_[l] = rd_f32(2 * d);
        rd_ql(d, d);                                         // Wq
        rd_ql(d, d);                                         // Wk
        rd_ql(d, d);                                         // Wv
        net.wo_off_[l] = rd_f32(d * d + d);
        net.ln2_off_[l] = rd_f32(2 * d);
        rd_ql(d, static_cast<int>(dff));                     // W1
        rd_ql(static_cast<int>(dff), d);                     // W2
    }
    net.lnP_ = rd_f32(2 * d);
    rd_ql(d, static_cast<int>(dpol));                        // Wfrom
    rd_ql(d, static_cast<int>(dpol));                        // Wto
    net.promo_ = rd_f32(4 * d + 4);
    net.lnV_ = rd_f32(2 * d);
    rd_ql(d, 128);                                           // V1
    net.v2_ = rd_f32(3 * 128 + 3);

    if (!f) return std::nullopt;
    return net;
}

NetQOutput NetQ::evaluate(const PositionState& pos) const {
    const int d = static_cast<int>(cfg_.d);
    const int heads = static_cast<int>(cfg_.heads);
    const int hd = d / heads;
    const int dff = static_cast<int>(cfg_.dff);
    const int dpol = static_cast<int>(cfg_.dpol);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // tokens (fp32 embeddings, shard piece-code convention)
    for (int sq = 0; sq < 64; ++sq) {
        std::uint8_t code = 0;
        for (int t = 0; t < PIECE_NB; ++t) {
            if (pos.pieces[WHITE][t] & SQUARE_BB[sq]) code = static_cast<std::uint8_t>(t + 1);
            if (pos.pieces[BLACK][t] & SQUARE_BB[sq]) code = static_cast<std::uint8_t>(t + 9);
        }
        for (int i = 0; i < d; ++i) {
            q_x[sq * d + i] = f_[emb_off_ + code * d + i] +
                              f_[sq_off_ + static_cast<std::size_t>(sq) * d + i] +
                              f_[side_off_ + static_cast<std::size_t>(pos.side_to_move) * d + i];
        }
    }

    for (std::uint32_t l = 0; l < cfg_.layers; ++l) {
        const std::size_t ln1 = ln1_off_[l], ln2 = ln2_off_[l], wo = wo_off_[l];
        const std::size_t wq = ql_from_ + l * 5;

        for (int s = 0; s < 64; ++s) {
            layernorm(&q_x[s * d], f_.data() + ln1, f_.data() + ln1 + d, q_vec, d);
            qlinear(q_vec, ql_[wq], &q_q[s * d]);
            qlinear(q_vec, ql_[wq + 1], &q_k[s * d]);
            qlinear(q_vec, ql_[wq + 2], &q_v[s * d]);
        }

        for (int head = 0; head < heads; ++head) {
            for (int s = 0; s < 64; ++s) {
                const float* qp = &q_q[s * d + head * hd];
                for (int t = 0; t < 64; ++t) {
                    const float* kp = &q_k[t * d + head * hd];
                    float dot = 0.0f;
                    for (int i = 0; i < hd; ++i) dot += qp[i] * kp[i];
                    q_att[s * 64 + t] = dot * scale;
                }
                softmax_inplace(&q_att[s * 64], 64);
                for (int i = 0; i < hd; ++i) {
                    float acc = 0.0f;
                    for (int t = 0; t < 64; ++t)
                        acc += q_att[s * 64 + t] * q_v[t * d + head * hd + i];
                    q_h[s * d + head * hd + i] = acc;
                }
            }
        }
        for (int s = 0; s < 64; ++s) {
            linear_f32(&q_h[s * d], f_.data() + wo, f_.data() + wo + d * d, q_vec, d, d);
            for (int i = 0; i < d; ++i) q_x[s * d + i] += q_vec[i];
        }
        for (int s = 0; s < 64; ++s) {
            layernorm(&q_x[s * d], f_.data() + ln2, f_.data() + ln2 + d, q_vec, d);
            qlinear(q_vec, ql_[wq + 3], q_wide);
            for (int i = 0; i < dff; ++i) q_wide[i] = gelu(q_wide[i]);
            qlinear(q_wide, ql_[wq + 4], q_vec);
            for (int i = 0; i < d; ++i) q_x[s * d + i] += q_vec[i];
        }
    }

    // heads
    NetQOutput out{};
    for (int sq = 0; sq < 64; ++sq)
        layernorm(&q_x[sq * d], f_.data() + lnP_, f_.data() + lnP_ + d, &q_h[sq * d], d);
    for (int i = 0; i < d; ++i) {
        float s = 0.0f;
        for (int sq = 0; sq < 64; ++sq) s += q_h[sq * d + i];
        q_pooled[i] = s / 64.0f;
    }
    layernorm(q_pooled, f_.data() + lnV_, f_.data() + lnV_ + d, q_hv, d);

    const std::size_t wfrom = ql_from_ + cfg_.layers * 5;
    const std::size_t wto = wfrom + 1;
    const std::size_t v1 = wto + 1;
    for (int sq = 0; sq < 64; ++sq) {
        qlinear(&q_h[sq * d], ql_[wfrom], &q_q[sq * d]);
        qlinear(&q_h[sq * d], ql_[wto], &q_k[sq * d]);
    }
    for (int u = 0; u < 64; ++u)
        for (int v = 0; v < 64; ++v) {
            float dot = 0.0f;
            for (int i = 0; i < dpol; ++i) dot += q_q[u * d + i] * q_k[v * d + i];
            out.scores[u * 64 + v] = dot / std::sqrt(static_cast<float>(dpol));
        }
    linear_f32(q_pooled, f_.data() + promo_, f_.data() + promo_ + 4 * d, out.promo_logit, d, 4);

    qlinear(q_hv, ql_[v1], q_vhid);
    for (int i = 0; i < 128; ++i) q_vhid[i] = gelu(q_vhid[i]);
    linear_f32(q_vhid, f_.data() + v2_, f_.data() + v2_ + 3 * 128, out.wdl, 128, 3);
    softmax_inplace(out.wdl, 3);
    return out;
}

float* NetQ::x_buf() {
    static thread_local float buf[64 * kMaxD];
    return buf;
}

}  // namespace lo::nn
