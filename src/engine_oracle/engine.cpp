#include "engine_oracle/engine.hpp"

#include <type_traits>

#include "movegen/movegen.hpp"
#include "tb/tb.hpp"

#include <array>
#include <limits>

namespace lo::engine {

namespace {

constexpr int MATERIAL[PIECE_NB] = {100, 320, 330, 500, 900, 0};

// Tables are white-perspective (a1 = index 0); black mirrors with sq ^ 56.

inline constexpr std::array<int, 64> PST_PAWN = [] {
    std::array<int, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        const int cf = f < 4 ? f : 7 - f;
        t[s] = r * 6 + (cf >= 2 ? 8 : 0);
    }
    return t;
}();

inline constexpr std::array<int, 64> PST_MINOR = [] {
    std::array<int, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        const int cf = f < 4 ? f : 7 - f;
        const int cr = r < 4 ? r : 7 - r;
        t[s] = (cf + cr) * 4;
    }
    return t;
}();

inline constexpr std::array<int, 64> PST_ROOK = [] {
    std::array<int, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        const int cf = f < 4 ? f : 7 - f;
        t[s] = (r == 6 ? 20 : 0) + (cf >= 2 ? 4 : 0);
    }
    return t;
}();

inline constexpr std::array<int, 64> PST_QUEEN = [] {
    std::array<int, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        const int cf = f < 4 ? f : 7 - f;
        const int cr = r < 4 ? r : 7 - r;
        t[s] = (cf + cr) * 2;
    }
    return t;
}();

inline constexpr std::array<int, 64> PST_KING = [] {
    std::array<int, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        const int cf = f < 4 ? f : 7 - f;
        const int cr = r < 4 ? r : 7 - r;
        t[s] = -5 * (cf + cr);
    }
    return t;
}();

inline const std::array<int, 64>* pst_for(PieceType t) {
    switch (t) {
        case PAWN: return &PST_PAWN;
        case KNIGHT:
        case BISHOP: return &PST_MINOR;
        case ROOK: return &PST_ROOK;
        case QUEEN: return &PST_QUEEN;
        default: return &PST_KING;
    }
}

bool insufficient_material(const PositionState& p) {
    if (p.pieces[WHITE][PAWN] | p.pieces[BLACK][PAWN] | p.pieces[WHITE][ROOK] |
        p.pieces[BLACK][ROOK] | p.pieces[WHITE][QUEEN] | p.pieces[BLACK][QUEEN])
        return false;
    const int minors = popcount(p.pieces[WHITE][KNIGHT] | p.pieces[WHITE][BISHOP] |
                                p.pieces[BLACK][KNIGHT] | p.pieces[BLACK][BISHOP]);
    return minors <= 1;
}

// Occurrences of `key` among prior game positions. Two prior occurrences means
// playing this move would produce the third.
std::uint32_t count_key(const std::vector<std::uint64_t>& h, std::uint64_t key) {
    std::uint32_t n = 0;
    for (std::uint64_t k : h)
        if (k == key) ++n;
    return n;
}

}  // namespace

int evaluate(const PositionState& p) {
    int score = 0;
    for (int c = 0; c < 2; ++c) {
        const int sign = (c == WHITE) ? 1 : -1;
        for (int t = 0; t < PIECE_NB; ++t) {
            const int* pst = pst_for(static_cast<PieceType>(t))->data();
            Bitboard b = p.pieces[c][t];
            while (b) {
                const Square s = static_cast<Square>(poplsb(b));
                const int idx = (c == WHITE) ? s : (s ^ 56);
                score += sign * (MATERIAL[t] + pst[idx]);
            }
        }
    }
    return score;
}

Move bestmove(const PositionState& p, const std::vector<std::uint64_t>& key_history) {
    MoveList ml;
    generate_legal(p, ml);
    if (ml.count == 0) return MOVE_NONE;

    // Tablebase root probe (same rationale as the net path).
    if (tb::available() && popcount(occupancy_all(p)) <= 5) {
        if (auto tbm = tb::probe_root(p)) {
            for (std::uint32_t i = 0; i < ml.count; ++i) {
                if ((ml.moves[i] & 0x0FFF) == (*tbm & 0x0FFF)) return *tbm;
            }
        }
    }

    Move best = ml.moves[0];
    int best_score = std::numeric_limits<int>::min();

    for (std::uint32_t i = 0; i < ml.count; ++i) {
        const Move m = ml.moves[i];
        const PositionState child = apply(p, m);

        int sc;
        if (insufficient_material(child)) {
            sc = 0;  // dead draw, cannot be avoided
        } else {
            const int e = evaluate(child);
            sc = (p.side_to_move == WHITE) ? e : -e;  // mover-relative

            // Symbolic gate: three-fold repetition and the 75-move automatic
            // draw. Winning, forbid the draw; losing, seek it; equal, it is
            // worth zero. Index term keeps selection deterministic.
            const bool child_draw =
                child.halfmove >= 150 || count_key(key_history, child.zobrist) >= 2;
            if (child_draw) {
                if (sc > 150)
                    sc = -1000000 + static_cast<int>(i);
                else if (sc < -150)
                    sc = 1000000 - static_cast<int>(i);
                else
                    sc = 0;
            }
        }

        if (sc > best_score) {
            best_score = sc;
            best = m;
        }
    }
    return best;
}

// Mirror-averaged policy (UCI MirrorAvg, default off).
//
// The policy head is indexed by absolute squares and the square embeddings are
// not mirror-equivariant, so the net's two mirrored views disagree
// systematically. Averaging them cancels part of that bias. Still one decision
// per move -- O(1) in search depth -- for one extra forward pass (~10 ms
// against a 15 s clock).
template <typename OutT, typename NetT>
struct MirrorView {
    OutT out{};
    PositionState pos{};
    nn::NetHistory hist{};

    // score the mirror image of move m in the mirrored evaluation
    float score_for(const Move m) const {
        const auto fu = static_cast<std::uint8_t>(move_from(m) ^ 7);
        const auto tu = static_cast<std::uint8_t>(move_to(m) ^ 7);
        float s = out.scores[static_cast<std::size_t>(fu) * 64 + tu];
        if (move_type(m) == MT_PROMO) s += out.promo_logit[move_promo(m)];
        return s;
    }
};

template <typename OutT, typename NetT>
void run_mirror_view(const PositionState& p, const NetT& net, const nn::NetHistory* hist,
                     MirrorView<OutT, NetT>& mv) {
    mv.pos = mirror_position(p);
    if (hist) {
        mv.hist = *hist;
        for (std::uint8_t i = 0; i < mv.hist.n; ++i) {
            mv.hist.from[i] = static_cast<std::uint8_t>(mv.hist.from[i] ^ 7);
            mv.hist.to[i] = static_cast<std::uint8_t>(mv.hist.to[i] ^ 7);
        }
        mv.out = net.evaluate(mv.pos, mv.hist);
    } else {
        mv.out = net.evaluate(mv.pos);
    }
}

template <typename NetT>
Move bestmove_net_impl(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                       const NetT& net, std::int64_t own_time_ms,
                       const nn::NetHistory* hist = nullptr,
                       bool mirror_avg = false) {
    // Low clock: the net call costs ~0.3-0.7 s; the PST path costs ~us.
    // Playing a weaker move beats losing on time.
    if (own_time_ms >= 0 && own_time_ms < 10000) return bestmove(p, key_history);

    // Tablebase root probe: provable endgame play beats the policy exactly
    // where a searchless engine is weakest (converting trivially won
    // endgames). The probe is DTZ-correct and respects the 50-move clock.
    if (tb::available() && popcount(occupancy_all(p)) <= 5) {
        if (auto tbm = tb::probe_root(p)) {
            MoveList tml;
            generate_legal(p, tml);
            for (std::uint32_t i = 0; i < tml.count; ++i) {
                if ((tml.moves[i] & 0x0FFF) == (*tbm & 0x0FFF)) return *tbm;
            }
        }
    }

    MoveList ml;
    generate_legal(p, ml);
    if (ml.count == 0) return MOVE_NONE;

    const auto out = hist ? net.evaluate(p, *hist) : net.evaluate(p);
    float wdl[3] = {out.wdl[0], out.wdl[1], out.wdl[2]};
    using OutT = std::remove_const_t<decltype(out)>;
    MirrorView<OutT, NetT> mv;
    if (mirror_avg) {
        run_mirror_view(p, net, hist, mv);
        for (int i = 0; i < 3; ++i) wdl[i] = 0.5f * (wdl[i] + mv.out.wdl[i]);
    }
    // WDL head is side-to-move POV.
    const float pwin = wdl[0], ploss = wdl[2];

    Move best = ml.moves[0];
    float best_score = -std::numeric_limits<float>::infinity();

    for (std::uint32_t i = 0; i < ml.count; ++i) {
        const Move m = ml.moves[i];
        float sc = out.scores[move_from(m) * 64 + move_to(m)];
        if (move_type(m) == MT_PROMO) sc += out.promo_logit[move_promo(m)];
        if (mirror_avg) sc += mv.score_for(m);

        const PositionState child = apply(p, m);
        if (insufficient_material(child)) {
            sc = -1e9f + static_cast<float>(i);  // dead draw, cannot be avoided
        } else {
            const bool child_draw =
                child.halfmove >= 150 ||
                count_key(key_history, child.zobrist) >= 2;
            if (child_draw) {
                // Winning: forbid the draw. Losing: seek it. Else: zero.
                if (pwin > 0.6f)
                    sc = -1e6f + static_cast<float>(i);
                else if (ploss > 0.6f)
                    sc = 1e6f - static_cast<float>(i);
                else
                    sc = 0.0f;
            }
        }

        if (sc > best_score) {
            best_score = sc;
            best = m;
        }
    }
        return best;
}

Move bestmove_net(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                  const nn::Net& net, std::int64_t own_time_ms) {
    return bestmove_net_impl(p, key_history, net, own_time_ms);
}

Move bestmove_net(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                  const nn::Net& net, std::int64_t own_time_ms, const nn::NetHistory& hist,
                  bool mirror_avg) {
    return bestmove_net_impl(p, key_history, net, own_time_ms, &hist, mirror_avg);
}

Move bestmove_net_q(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                    const nn::NetQ& net, std::int64_t own_time_ms) {
    return bestmove_net_impl(p, key_history, net, own_time_ms);
}

Move bestmove_net_q(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                    const nn::NetQ& net, std::int64_t own_time_ms, const nn::NetHistory& hist,
                    bool mirror_avg) {
    return bestmove_net_impl(p, key_history, net, own_time_ms, &hist, mirror_avg);
}

}  // namespace lo::engine
