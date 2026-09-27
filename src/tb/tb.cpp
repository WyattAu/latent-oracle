#include "tb/tb.hpp"

#include "tbprobe.h"

#include <cstring>

namespace lo::tb {

namespace {

bool g_init = false;

std::optional<Move> from_tb_move(unsigned res) {
    const unsigned from = TB_GET_FROM(res);
    const unsigned to = TB_GET_TO(res);
    const unsigned promo = TB_GET_PROMOTES(res);
    const bool ep = TB_GET_EP(res) != 0;

    // Fathom promo order (1=Q 2=R 3=B 4=N) -> ours (0=N 1=B 2=R 3=Q).
    int promo_ours = 255;
    switch (promo) {
        case TB_PROMOTES_QUEEN: promo_ours = 3; break;
        case TB_PROMOTES_ROOK: promo_ours = 2; break;
        case TB_PROMOTES_BISHOP: promo_ours = 1; break;
        case TB_PROMOTES_KNIGHT: promo_ours = 0; break;
        default: break;
    }
    const MoveType type = ep       ? MT_EN_PASSANT
                          : promo_ours != 255 ? MT_PROMO
                                              : MT_NORMAL;
    return make_move(static_cast<Square>(from), static_cast<Square>(to), type,
                     promo_ours == 255 ? 0 : promo_ours);
}

}  // namespace

bool init(const std::string& path) {
    g_init = tb_init(path.c_str());
    return g_init;
}

bool available() { return g_init; }

std::optional<Move> probe_root(const PositionState& pos) {
    if (!g_init || pos.castling != 0) return std::nullopt;

    const Bitboard white = occupancy(pos, WHITE);
    const Bitboard black = occupancy(pos, BLACK);
    if (popcount(white | black) > 5) return std::nullopt;

    unsigned results[TB_MAX_MOVES];
    const unsigned res = tb_probe_root(
        white, black,
        pos.pieces[WHITE][KING] | pos.pieces[BLACK][KING],
        pos.pieces[WHITE][QUEEN] | pos.pieces[BLACK][QUEEN],
        pos.pieces[WHITE][ROOK] | pos.pieces[BLACK][ROOK],
        pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][BISHOP],
        pos.pieces[WHITE][KNIGHT] | pos.pieces[BLACK][KNIGHT],
        pos.pieces[WHITE][PAWN] | pos.pieces[BLACK][PAWN],
        pos.halfmove,
        0,  // castling: probing requires no rights (we gate above)
        pos.ep_square == NO_EP ? 0u : static_cast<unsigned>(pos.ep_square),
        pos.side_to_move == WHITE,
        results);
    if (res == TB_RESULT_FAILED || res == TB_RESULT_STALEMATE ||
        res == TB_RESULT_CHECKMATE)
        return std::nullopt;

    return from_tb_move(res);
}

}  // namespace lo::tb
