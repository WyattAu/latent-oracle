#include "position.hpp"

#include <cstdlib>

namespace lo {

std::optional<PositionState> parse_fen(std::string_view fen) {
    // Tokenize by whitespace.
    std::string_view fields[8];
    int nfields = 0;
    std::size_t i = 0;
    while (i < fen.size() && nfields < 8) {
        while (i < fen.size() && (fen[i] == ' ' || fen[i] == '\t')) ++i;
        if (i >= fen.size()) break;
        const std::size_t start = i;
        while (i < fen.size() && fen[i] != ' ' && fen[i] != '\t') ++i;
        fields[nfields++] = fen.substr(start, i - start);
    }
    if (nfields < 2) return std::nullopt;

    PositionState p{};
    p.ep_square = NO_EP;  // zero-init would alias ep square a1 (latent bug)

    // 1. Piece placement: ranks 8 -> 1, files a -> h.
    {
        int r = 7, f = 0;
        for (const char ch : fields[0]) {
            if (ch == '/') {
                if (f != 8 || r == 0) return std::nullopt;
                --r;
                f = 0;
                continue;
            }
            Color c = WHITE;
            PieceType t;
            switch (ch) {
                case 'P': t = PAWN; break;
                case 'N': t = KNIGHT; break;
                case 'B': t = BISHOP; break;
                case 'R': t = ROOK; break;
                case 'Q': t = QUEEN; break;
                case 'K': t = KING; break;
                case 'p': c = BLACK; t = PAWN; break;
                case 'n': c = BLACK; t = KNIGHT; break;
                case 'b': c = BLACK; t = BISHOP; break;
                case 'r': c = BLACK; t = ROOK; break;
                case 'q': c = BLACK; t = QUEEN; break;
                case 'k': c = BLACK; t = KING; break;
                default:
                    if (ch < '1' || ch > '8') return std::nullopt;
                    f += ch - '0';
                    continue;
            }
            if (f > 7) return std::nullopt;
            const Square sq = make_square(f, r);
            p.pieces[c][t] |= SQUARE_BB[sq];
            p.zobrist ^= ZOBRIST.piece[c][t][sq];
            ++f;
        }
        if (r != 0 || f != 8) return std::nullopt;
    }

    // 2. Side to move.
    if (fields[1] == "w")
        p.side_to_move = WHITE;
    else if (fields[1] == "b")
        p.side_to_move = BLACK;
    else
        return std::nullopt;

    // 3. Castling rights.
    if (nfields >= 3 && fields[2] != "-") {
        for (const char ch : fields[2]) {
            switch (ch) {
                case 'K': p.castling |= CR_WK; break;
                case 'Q': p.castling |= CR_WQ; break;
                case 'k': p.castling |= CR_BK; break;
                case 'q': p.castling |= CR_BQ; break;
                default: return std::nullopt;
            }
        }
    }
    p.zobrist ^= ZOBRIST.castle[p.castling];

    // 4. En passant target, normalized to capturable-only.
    if (nfields >= 4 && fields[3] != "-") {
        const Square ep = square_from_name(fields[3]);
        if (static_cast<int>(ep) < 0) return std::nullopt;
        const Color us = static_cast<Color>(p.side_to_move);
        if (PAWN_ATTACKS[us][ep] & p.pieces[~us][PAWN]) {
            p.ep_square = static_cast<std::uint8_t>(ep);
            p.zobrist ^= ZOBRIST.ep[ep & 7];
        }
    }

    // 5./6. Halfmove and fullmove counters (optional).
    if (nfields >= 5) {
        p.halfmove = static_cast<std::uint8_t>(std::atoi(std::string(fields[4]).c_str()));
    }
    if (nfields >= 6) {
        p.fullmove = static_cast<std::uint16_t>(std::atoi(std::string(fields[5]).c_str()));
    }
    if (p.fullmove == 0) p.fullmove = 1;

    return p;
}

bool attacked(const PositionState& p, Square s, Color by) {
    // A `by`-pawn attacks s iff it stands on a square matching the reverse
    // pawn-attack pattern of s.
    if (PAWN_ATTACKS[~by][s] & p.pieces[by][PAWN]) return true;
    if (KNIGHT_ATTACKS[s] & p.pieces[by][KNIGHT]) return true;
    if (KING_ATTACKS[s] & p.pieces[by][KING]) return true;

    const Bitboard occ = occupancy_all(p);
    if (bishop_attacks(s, occ) & (p.pieces[by][BISHOP] | p.pieces[by][QUEEN])) return true;
    if (rook_attacks(s, occ) & (p.pieces[by][ROOK] | p.pieces[by][QUEEN])) return true;
    return false;
}

PositionState apply(const PositionState& p, Move m) {
    PositionState n = p;
    const Color us = static_cast<Color>(p.side_to_move);
    const Color them = ~us;
    const Square from = move_from(m);
    const Square to = move_to(m);
    const MoveType type = move_type(m);
    const PieceType pt = piece_type_at(p, from, us);
    const Bitboard toBB = SQUARE_BB[to];

    std::uint64_t key = p.zobrist ^ ZOBRIST.piece[us][pt][from];
    if (p.ep_square != NO_EP) key ^= ZOBRIST.ep[p.ep_square & 7];
    key ^= ZOBRIST.castle[p.castling];

    n.pieces[us][pt] &= ~SQUARE_BB[from];
    n.halfmove = (pt == PAWN) ? 0 : static_cast<std::uint8_t>(p.halfmove + 1);

    // Captures (en passant handled separately; its target square is empty).
    for (int t = 0; t < PIECE_NB; ++t) {
        if (p.pieces[them][t] & toBB) {
            n.pieces[them][t] &= ~toBB;
            key ^= ZOBRIST.piece[them][t][to];
            n.halfmove = 0;
            break;
        }
    }

    PieceType placed = pt;
    if (type == MT_PROMO) placed = static_cast<PieceType>(KNIGHT + move_promo(m));

    n.pieces[us][placed] |= toBB;
    key ^= ZOBRIST.piece[us][placed][to];

    n.ep_square = NO_EP;
    switch (type) {
        case MT_EN_PASSANT: {
            const Square vic = static_cast<Square>(static_cast<int>(to) + (us == WHITE ? -8 : 8));
            n.pieces[them][PAWN] &= ~SQUARE_BB[vic];
            key ^= ZOBRIST.piece[them][PAWN][vic];
            n.halfmove = 0;
            break;
        }
        case MT_CASTLE: {
            const Square rf = (to > from) ? static_cast<Square>(to + 1) : static_cast<Square>(to - 2);
            const Square rt = (to > from) ? static_cast<Square>(to - 1) : static_cast<Square>(to + 1);
            n.pieces[us][ROOK] &= ~SQUARE_BB[rf];
            n.pieces[us][ROOK] |= SQUARE_BB[rt];
            key ^= ZOBRIST.piece[us][ROOK][rf] ^ ZOBRIST.piece[us][ROOK][rt];
            break;
        }
        default: {
            if (pt == PAWN && (to == from + 16 || from == to + 16)) {
                const Square ep =
                    static_cast<Square>(static_cast<int>(from) + (us == WHITE ? 8 : -8));
                if (PAWN_ATTACKS[us][ep] & p.pieces[them][PAWN]) {
                    n.ep_square = static_cast<std::uint8_t>(ep);
                    key ^= ZOBRIST.ep[ep & 7];
                }
            }
            break;
        }
    }

    const std::uint8_t new_castling = p.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
    key ^= ZOBRIST.castle[new_castling];
    n.castling = new_castling;

    n.side_to_move = static_cast<std::uint8_t>(them);
    key ^= ZOBRIST.side;
    if (us == BLACK) n.fullmove = static_cast<std::uint16_t>(p.fullmove + 1);

    n.zobrist = key;
    return n;
}

}  // namespace lo
