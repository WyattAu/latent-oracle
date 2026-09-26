#include "movegen/movegen.hpp"

namespace lo {

namespace {

inline void add_pawn_moves(MoveList& out, Square from, Square to, bool promotion) {
    if (promotion) {
        out.push(make_move(from, to, MT_PROMO, 3));  // queen
        out.push(make_move(from, to, MT_PROMO, 0));  // knight
        out.push(make_move(from, to, MT_PROMO, 2));  // rook
        out.push(make_move(from, to, MT_PROMO, 1));  // bishop
    } else {
        out.push(make_move(from, to));
    }
}

void generate_pawn(const PositionState& p, MoveList& out) {
    const Color us = static_cast<Color>(p.side_to_move);
    const Color them = ~us;
    const Bitboard all = occupancy_all(p);
    const Bitboard enemy = occupancy(p, them);
    const int dir = (us == WHITE) ? 8 : -8;
    const int promo_rank = (us == WHITE) ? 7 : 0;
    const int start_rank = (us == WHITE) ? 1 : 6;

    Bitboard b = p.pieces[us][PAWN];
    while (b) {
        const Square s = static_cast<Square>(poplsb(b));
        const int f = file_of(s);

        // Pushes.
        const Square to = static_cast<Square>(static_cast<int>(s) + dir);
        if (!(all & SQUARE_BB[to])) {
            add_pawn_moves(out, s, to, rank_of(to) == promo_rank);
            if (rank_of(s) == start_rank) {
                const Square to2 = static_cast<Square>(static_cast<int>(to) + dir);
                if (!(all & SQUARE_BB[to2])) out.push(make_move(s, to2));
            }
        }

        // Captures (including en passant).
        for (int df = -1; df <= 1; df += 2) {
            const int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            const Square t = static_cast<Square>(static_cast<int>(to) + df);
            if (enemy & SQUARE_BB[t]) {
                add_pawn_moves(out, s, t, rank_of(t) == promo_rank);
            } else if (p.ep_square != NO_EP && t == static_cast<Square>(p.ep_square)) {
                out.push(make_move(s, t, MT_EN_PASSANT));
            }
        }
    }
}

template <PieceType PT>
void generate_step(const PositionState& p, MoveList& out) {
    const Color us = static_cast<Color>(p.side_to_move);
    const Bitboard own = occupancy(p, us);
    const auto& table = (PT == KNIGHT) ? KNIGHT_ATTACKS : KING_ATTACKS;

    Bitboard b = p.pieces[us][PT];
    while (b) {
        const Square s = static_cast<Square>(poplsb(b));
        Bitboard t = table[s] & ~own;
        while (t) {
            const Square to = static_cast<Square>(poplsb(t));
            out.push(make_move(s, to));
        }
    }
}

template <PieceType PT>
void generate_slider(const PositionState& p, MoveList& out) {
    const Color us = static_cast<Color>(p.side_to_move);
    const Bitboard own = occupancy(p, us);
    const Bitboard all = own | occupancy(p, ~us);

    Bitboard b = p.pieces[us][PT];
    while (b) {
        const Square s = static_cast<Square>(poplsb(b));
        Bitboard atk;
        if constexpr (PT == QUEEN)
            atk = rook_attacks(s, all) | bishop_attacks(s, all);
        else if constexpr (PT == ROOK)
            atk = rook_attacks(s, all);
        else
            atk = bishop_attacks(s, all);
        atk &= ~own;
        while (atk) {
            const Square to = static_cast<Square>(poplsb(atk));
            out.push(make_move(s, to));
        }
    }
}

void generate_castles(const PositionState& p, MoveList& out) {
    const Color us = static_cast<Color>(p.side_to_move);
    const Color them = ~us;
    const Bitboard all = occupancy_all(p);

    if (us == WHITE) {
        if ((p.castling & CR_WK) && (p.pieces[WHITE][KING] & SQUARE_BB[SQ_E1]) &&
            (p.pieces[WHITE][ROOK] & SQUARE_BB[SQ_H1]) && !(all & 0x0000000000000060ULL) &&
            !attacked(p, SQ_E1, them) && !attacked(p, SQ_F1, them)) {
            out.push(make_move(SQ_E1, SQ_G1, MT_CASTLE));
        }
        if ((p.castling & CR_WQ) && (p.pieces[WHITE][KING] & SQUARE_BB[SQ_E1]) &&
            (p.pieces[WHITE][ROOK] & SQUARE_BB[SQ_A1]) && !(all & 0x000000000000000EULL) &&
            !attacked(p, SQ_E1, them) && !attacked(p, SQ_D1, them)) {
            out.push(make_move(SQ_E1, SQ_C1, MT_CASTLE));
        }
    } else {
        if ((p.castling & CR_BK) && (p.pieces[BLACK][KING] & SQUARE_BB[SQ_E8]) &&
            (p.pieces[BLACK][ROOK] & SQUARE_BB[SQ_H8]) &&
            !(all & 0x6000000000000000ULL) && !attacked(p, SQ_E8, them) &&
            !attacked(p, SQ_F8, them)) {
            out.push(make_move(SQ_E8, SQ_G8, MT_CASTLE));
        }
        if ((p.castling & CR_BQ) && (p.pieces[BLACK][KING] & SQUARE_BB[SQ_E8]) &&
            (p.pieces[BLACK][ROOK] & SQUARE_BB[SQ_A8]) &&
            !(all & 0x0E00000000000000ULL) && !attacked(p, SQ_E8, them) &&
            !attacked(p, SQ_D8, them)) {
            out.push(make_move(SQ_E8, SQ_C8, MT_CASTLE));
        }
    }
}

}  // namespace

void generate_pseudo(const PositionState& p, MoveList& out) {
    generate_pawn(p, out);
    generate_step<KNIGHT>(p, out);
    generate_step<KING>(p, out);
    generate_slider<BISHOP>(p, out);
    generate_slider<ROOK>(p, out);
    generate_slider<QUEEN>(p, out);
    generate_castles(p, out);
}

void generate_legal(const PositionState& p, MoveList& out) {
    MoveList tmp;
    generate_pseudo(p, tmp);

    const Color us = static_cast<Color>(p.side_to_move);
    const Color them = ~us;
    for (std::uint32_t i = 0; i < tmp.count; ++i) {
        const Move m = tmp.moves[i];
        const PositionState child = apply(p, m);
        const Square ksq = static_cast<Square>(lsb(child.pieces[us][KING]));
        if (!attacked(child, ksq, them)) out.push(m);
    }
}

}  // namespace lo
