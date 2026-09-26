#pragma once

// Position lifecycle: FEN parsing, attack detection, and make-move (copy-make;
// PositionState is 128 bytes, so copying is cheaper than undo bookkeeping).

#include "movegen/attack.hpp"
#include "types.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace lo {

inline constexpr std::string_view STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Parses a FEN string. The ep field is normalized: the ep square is retained
// only when an enemy pawn could actually capture, which keeps Zobrist keys
// exact for repetition detection.
std::optional<PositionState> parse_fen(std::string_view fen);

// Type of the piece of color c standing on s. Requires occupancy (legal input).
inline PieceType piece_type_at(const PositionState& p, Square s, Color c) {
    const Bitboard b = SQUARE_BB[s];
    for (int t = 0; t < PIECE_NB; ++t)
        if (p.pieces[c][t] & b) return static_cast<PieceType>(t);
    return KING;  // unreachable for legal positions
}

// Is square s attacked by any piece of color `by`?
bool attacked(const PositionState& p, Square s, Color by);

// Copy-make. Returns the child state; Zobrist key updated incrementally.
PositionState apply(const PositionState& p, Move m);

inline std::string move_to_uci(Move m) {
    std::string s = square_name(move_from(m)) + square_name(move_to(m));
    if (move_type(m) == MT_PROMO) {
        constexpr char kPromo[4] = {'n', 'b', 'r', 'q'};
        s += kPromo[move_promo(m)];
    }
    return s;
}

}  // namespace lo
