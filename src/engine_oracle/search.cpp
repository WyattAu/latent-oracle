#include "engine_oracle/search.hpp"

#include "engine_oracle/engine.hpp"
#include "movegen/movegen.hpp"
#include "tb/tb.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <vector>

namespace lo::uci {
extern std::atomic<bool> g_stop;
}
// ============================================================================
// TODO(known-bug, next session): the AB root misreads the position.
// Symptom: on "position startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
// ab() sees in_check=1 (false) and only 20 legal moves (true: 30); it
// returns -MATE-1 instantly. The policy path (bestmove_net_q) on the SAME
// s.root plays g1f3 correctly, so set_position is fine — the bug is in how
// ab()/qsearch() consume the PositionState (likely the Color casts of
// side_to_move, the king_sq() occupancy math, or the rep_stack). Debug:
// print b.fen()-equivalent from ab() at ply 0 and diff vs set_position.
// Everything else works: ID loop, TT, qsearch, movegen integration.
// ===========================================================================

namespace lo::engine {
namespace {

constexpr int MATE = 30000;
constexpr int MATE_IN_MAX = MATE - 256;

template <typename OutT>
float value_cp(const OutT& o) {
    return ((o.wdl[0] + 0.5f * o.wdl[1]) - 0.5f) * 2.0f * 1000.0f;
}

enum : std::uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };
struct TTEntry {
    std::uint64_t key = 0;
    std::int16_t score = 0;
    std::int16_t depth = -1;
    std::uint8_t flag = TT_UPPER;
    std::uint32_t move = 0;
};
std::unordered_map<std::uint64_t, TTEntry> g_tt;

struct SearchCtx {
    const nn::NetQ* net = nullptr;
    std::uint64_t nodes = 0;
    std::uint64_t node_cap = 0;
    std::chrono::steady_clock::time_point deadline;
    std::vector<std::uint64_t> rep_stack;
};

bool time_up(const SearchCtx& ctx) {
    if (ctx.nodes >= ctx.node_cap) return true;
    return (ctx.nodes & 2047) == 0 &&
           std::chrono::steady_clock::now() > ctx.deadline;
}

bool is_draw(const PositionState& p, const SearchCtx& ctx) {
    if (p.halfmove >= 100) return true;
    int reps = 0;
    for (auto k : ctx.rep_stack) reps += (k == p.zobrist);
    // 3-fold: the stack includes the current position itself, so the
    // current occurrence counts — a first visit is NOT a repetition.
    return reps >= 3;
}

Square king_sq(const PositionState& p, Color c) {
    return Square(__builtin_ctzll(occupancy(p, c) & p.pieces[c][KING]));
}

template <typename NetT>
int qsearch(PositionState p, int alpha, int beta, int ply, SearchCtx& ctx) {
    const Color us = static_cast<Color>(p.side_to_move);
    const Color them = static_cast<Color>(p.side_to_move ^ 1); // FIX: ~uint8_t promotes to -1/-2, not the XOR
    ++ctx.nodes;
    if (time_up(ctx)) return alpha;
    const auto out = ctx.net->evaluate(p);
    int stand = static_cast<int>(value_cp(out));
    if (ply >= MATE_IN_MAX) return stand;
    if (stand >= beta) return stand;
    if (stand > alpha) alpha = stand;

    MoveList ml;
    generate_legal(p, ml);
    MoveList caps;
    for (std::uint32_t i = 0; i < ml.count; ++i)
        if (move_type(ml.moves[i]) == MT_PROMO ||
            (occupancy(p, them) & SQUARE_BB[move_to(ml.moves[i])]))
            caps.push(ml.moves[i]);
    ml = caps;

    int scores[256];
    for (std::uint32_t i = 0; i < ml.count; ++i) {
        PieceType vic = piece_type_at(p, move_to(ml.moves[i]), them);
        PieceType att = piece_type_at(p, move_from(ml.moves[i]), us);
        scores[i] = 10 * static_cast<int>(vic) - static_cast<int>(att) + 1000;
    }
    for (std::uint32_t a = 0; a + 1 < ml.count; ++a) {
        std::uint32_t best_i = a;
        for (std::uint32_t j = a + 1; j < ml.count; ++j)
            if (scores[j] > scores[a]) best_i = j;
        if (best_i != a) {
            std::swap(ml.moves[a], ml.moves[best_i]);
            std::swap(scores[a], scores[best_i]);
        }
    }
    for (std::uint32_t i = 0; i < ml.count; ++i) {
        PositionState child = apply(p, ml.moves[i]);
        int score = -qsearch<NetT>(child, -beta, -alpha, ply + 1, ctx);
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

template <typename NetT>
int ab(PositionState p, int depth, int alpha, int beta, int ply, SearchCtx& ctx) {
    const bool root = (ply == 0);
    const Color us = static_cast<Color>(p.side_to_move);
    const Color them = static_cast<Color>(p.side_to_move ^ 1); // FIX: ~uint8_t promotes to -1/-2, not the XOR
    const bool in_check = attacked(p, king_sq(p, static_cast<Color>(p.side_to_move)), static_cast<Color>(p.side_to_move ^ 1));
    if (ply > 0 && (is_draw(p, ctx) || ply >= MATE_IN_MAX))
        return is_draw(p, ctx) ? 0 : static_cast<int>(value_cp(ctx.net->evaluate(p)));

    ++ctx.nodes;
    if (ply > 0 && time_up(ctx))
        return static_cast<int>(value_cp(ctx.net->evaluate(p)));

    TTEntry* tte = nullptr;
    if (auto it = g_tt.find(p.zobrist); it != g_tt.end()) {
        tte = &it->second;
        if (tte->depth >= depth && ply > 0) {
            if (tte->flag == TT_EXACT) return tte->score;
            if (tte->flag == TT_LOWER && tte->score >= beta) return tte->score;
            if (tte->flag == TT_UPPER && tte->score <= alpha) return tte->score;
        }
    }

    if (depth <= 0) {
        if (!in_check) return qsearch<NetT>(p, alpha, beta, ply, ctx);
        depth = 1;
    }

    MoveList ml;
    generate_legal(p, ml);
    if (ply == 0 || ctx.nodes < 3)
        std::fprintf(stderr, "[ab] ply=%d depth=%d ml.count=%u in_check=%d\n",
                     ply, depth, ml.count, in_check);
    if (ml.count == 0)
        return in_check ? -MATE + ply : 0;

    int order[256];
    for (std::uint32_t i = 0; i < ml.count; ++i) {
        order[i] = 0;
        if (tte && tte->move == ml.moves[i]) order[i] = 1 << 20;
        if (move_type(ml.moves[i]) == MT_PROMO) order[i] += 900;
        PieceType vic = piece_type_at(p, move_to(ml.moves[i]), them);
        if (vic != KING && (occupancy(p, them) & SQUARE_BB[move_to(ml.moves[i])]))
            order[i] += 10 * static_cast<int>(vic) + 500;
    }
    for (std::uint32_t a = 0; a + 1 < ml.count; ++a) {
        std::uint32_t best_i = a;
        for (std::uint32_t j = a + 1; j < ml.count; ++j)
            if (order[j] > order[a]) best_i = j;
        if (best_i != a) {
            std::swap(ml.moves[a], ml.moves[best_i]);
            std::swap(order[a], order[best_i]);
        }
    }

    int best = -MATE - 1;
    Move best_move = ml.moves[0];
    int a0 = alpha;
    std::uint8_t flag = TT_UPPER;

    for (std::uint32_t i = 0; i < ml.count; ++i) {
        Move m = ml.moves[i];
        PositionState child = apply(p, m);
        if (attacked(child, king_sq(child, us), them))
            continue;
        ctx.rep_stack.push_back(child.zobrist);
        int score;
        int red = (depth >= 3 && i >= 4 && !in_check &&
                   piece_type_at(p, move_from(m), us) != PAWN) ? 1 : 0;
        if (root && i == 0)
            score = -ab<NetT>(child, depth - 1, -beta, -alpha, ply + 1, ctx);
        else {
            score = -ab<NetT>(child, depth - 1 - red, -alpha - 1, -alpha, ply + 1, ctx);
            if (score > alpha)
                score = -ab<NetT>(child, depth - 1, -beta, -alpha, ply + 1, ctx);
        }
        ctx.rep_stack.pop_back();
        if (time_up(ctx)) return alpha;

        if (score > best) {
            best = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
                flag = TT_EXACT;
            }
            if (alpha >= beta) { flag = TT_LOWER; break; }
        }
    }

    if (!tte && g_tt.size() < 2'000'000) {
        auto res = g_tt.emplace(p.zobrist, TTEntry{});
        tte = &res.first->second;
        tte->key = p.zobrist;
    }
    if (tte) {
        tte->score = static_cast<std::int16_t>(std::clamp(best, -MATE, MATE));
        tte->depth = static_cast<std::int16_t>(depth);
        tte->flag = flag;
        tte->move = best_move;
    }
    return best;
}

}  // namespace

Move bestmove_ab(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                 const nn::NetQ& net, std::int64_t own_time_ms,
                 std::uint64_t ab_nodes, std::int64_t ab_time_ms) {
    if (own_time_ms >= 0 && own_time_ms < 10000) return bestmove(p, key_history);
    if (tb::available() && popcount(occupancy_all(p)) <= 5) {
        if (auto tbm = tb::probe_root(p)) {
            MoveList tml;
            generate_legal(p, tml);
            for (std::uint32_t i = 0; i < tml.count; ++i)
                if ((tml.moves[i] & 0x0FFF) == (*tbm & 0x0FFF)) return *tbm;
        }
    }

    g_tt.clear();
    SearchCtx ctx;
    ctx.net = &net;
    ctx.node_cap = ab_nodes;
    ctx.deadline = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(std::min<std::int64_t>(ab_time_ms, 3000));
    ctx.rep_stack.assign(key_history.begin(), key_history.end());

    MoveList ml;
    generate_legal(p, ml);
    if (ml.count == 0) return MOVE_NONE;

    Move best = ml.moves[0];
    for (int depth = 1; depth <= 64; ++depth) {
        int score = ab<nn::NetQ>(p, depth, -MATE - 1, MATE + 1, 0, ctx);
        std::fprintf(stderr, "[ab] depth=%d score=%d best=%s nodes=%llu\n",
                     depth, score, move_to_uci(best).c_str(),
                     static_cast<unsigned long long>(ctx.nodes));
        if (time_up(ctx)) break;
        if (auto it = g_tt.find(p.zobrist); it != g_tt.end() && it->second.move)
            best = static_cast<Move>(it->second.move);
        if (std::abs(score) >= MATE_IN_MAX) break;
    }
    return best;
}

}  // namespace lo::engine
