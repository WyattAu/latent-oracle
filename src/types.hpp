#pragma once

// Core chess data types: squares, bitboards, move encoding, position state,
// deterministic Zobrist keys, and constexpr attack primitives.
//
// Layout contract: PositionState is exactly 128 bytes (two L1 cache lines),
// pointer-free, and carries no dynamic indirection. Occupancy bitboards are
// derived on demand rather than stored (plan of record: corrected layout).

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace lo {

using Bitboard = std::uint64_t;

// ---------------------------------------------------------------------------
// Colors, piece types, squares
// ---------------------------------------------------------------------------

enum Color : int { WHITE = 0, BLACK = 1 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    PIECE_NB = 6
};

// Rank-major with a1 = 0, h1 = 7, a2 = 8, ..., h8 = 63.
enum Square : int {
    SQ_A1 = 0, SQ_B1 = 1, SQ_C1 = 2, SQ_D1 = 3, SQ_E1 = 4, SQ_F1 = 5, SQ_G1 = 6, SQ_H1 = 7,
    SQ_A8 = 56, SQ_B8 = 57, SQ_C8 = 58, SQ_D8 = 59, SQ_E8 = 60, SQ_F8 = 61, SQ_G8 = 62, SQ_H8 = 63
};

constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int f, int r) { return Square(r * 8 + f); }

inline constexpr std::array<Bitboard, 64> SQUARE_BB = [] {
    std::array<Bitboard, 64> b{};
    for (int i = 0; i < 64; ++i)
        b[static_cast<std::size_t>(i)] = 1ULL << i;
    return b;
}();

inline constexpr Bitboard FILE_A = 0x0101010101010101ULL;
inline constexpr Bitboard FILE_H = 0x8080808080808080ULL;
inline constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
inline constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

// ---------------------------------------------------------------------------
// Bit primitives
// ---------------------------------------------------------------------------

constexpr int lsb(Bitboard b) { return __builtin_ctzll(b); }
constexpr int msb(Bitboard b) { return 63 - __builtin_clzll(b); }
constexpr int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline int poplsb(Bitboard& b) {
    const int s = lsb(b);
    b &= b - 1;
    return s;
}

// ---------------------------------------------------------------------------
// Move encoding: [5:0] from | [11:6] to | [13:12] promo | [15:14] type
// Promo piece order: 0 = knight, 1 = bishop, 2 = rook, 3 = queen.
// ---------------------------------------------------------------------------

enum MoveType : int { MT_NORMAL = 0, MT_EN_PASSANT = 1, MT_CASTLE = 2, MT_PROMO = 3 };

using Move = std::uint16_t;
constexpr Move MOVE_NONE = 0;

constexpr Move make_move(Square from, Square to, MoveType t = MT_NORMAL, int promo = 0) {
    return static_cast<Move>(from | (to << 6) | (promo << 12) | (t << 14));
}
constexpr Square move_from(Move m) { return static_cast<Square>(m & 0x3F); }
constexpr Square move_to(Move m) { return static_cast<Square>((m >> 6) & 0x3F); }
constexpr int move_promo(Move m) { return (m >> 12) & 3; }
constexpr MoveType move_type(Move m) { return static_cast<MoveType>((m >> 14) & 3); }

struct MoveList {
    std::uint32_t count = 0;
    Move moves[256];

    void push(Move m) { moves[count++] = m; }
};
static_assert(sizeof(MoveList) == 4 + 256 * sizeof(Move), "MoveList layout drift");

// ---------------------------------------------------------------------------
// PositionState: exactly two cache lines
// ---------------------------------------------------------------------------

constexpr std::uint8_t NO_EP = 0xFF;
enum CastlingRights : int { CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8 };

struct alignas(64) PositionState {
    std::uint64_t pieces[2][PIECE_NB];  // [color][piece type], a1 = bit 0
    std::uint64_t zobrist;              // incremental hash (piece/side/castle/ep)
    std::uint8_t ep_square;             // target square or NO_EP (only set when capturable)
    std::uint8_t castling;
    std::uint8_t side_to_move;          // Color
    std::uint8_t halfmove;              // 50-move counter (halfmoves)
    std::uint16_t fullmove;
    std::uint8_t pad[18];               // explicit: struct closes at 128 bytes exactly
};

static_assert(sizeof(PositionState) == 128, "PositionState must be exactly 128 bytes");
static_assert(alignof(PositionState) == 64, "PositionState must be cache-line aligned");

inline Bitboard occupancy(const PositionState& p, Color c) {
    Bitboard b = 0;
    for (int t = 0; t < PIECE_NB; ++t)
        b |= p.pieces[c][t];
    return b;
}
inline Bitboard occupancy_all(const PositionState& p) {
    return occupancy(p, WHITE) | occupancy(p, BLACK);
}

// ---------------------------------------------------------------------------
// Zobrist keys: deterministic (splitmix64), stable across builds and runs
// ---------------------------------------------------------------------------

struct ZobristKeys {
    std::uint64_t piece[2][PIECE_NB][64];
    std::uint64_t castle[16];
    std::uint64_t ep[8];
    std::uint64_t side;
};

consteval ZobristKeys make_zobrist() {
    ZobristKeys z{};
    std::uint64_t s = 0x0123456789ABCDE7ULL;
    auto next = [&s]() -> std::uint64_t {
        s += 0x9E3779B97F4A7C15ULL;
        std::uint64_t x = s;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    };
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < PIECE_NB; ++t)
            for (int sq = 0; sq < 64; ++sq)
                z.piece[c][t][sq] = next();
    for (int i = 0; i < 16; ++i)
        z.castle[i] = next();
    for (int i = 0; i < 8; ++i)
        z.ep[i] = next();
    z.side = next();
    return z;
}
inline constexpr ZobristKeys ZOBRIST = make_zobrist();

// Castling-rights update masks: rights &= MASK[from] & MASK[to].
inline constexpr std::array<std::uint8_t, 64> CASTLE_MASK = [] {
    std::array<std::uint8_t, 64> m{};
    m.fill(0xF);
    m[SQ_A1] = 0xF & ~CR_WQ;
    m[SQ_H1] = 0xF & ~CR_WK;
    m[SQ_E1] = 0xF & ~(CR_WK | CR_WQ);
    m[SQ_A8] = 0xF & ~CR_BQ;
    m[SQ_H8] = 0xF & ~CR_BK;
    m[SQ_E8] = 0xF & ~(CR_BK | CR_BQ);
    return m;
}();

// ---------------------------------------------------------------------------
// constexpr step-piece attack tables
// ---------------------------------------------------------------------------

// PAWN_ATTACKS[c][s]: squares attacked by a c-pawn standing on s.
inline constexpr std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS = [] {
    std::array<std::array<Bitboard, 64>, 2> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        for (int c = 0; c < 2; ++c) {
            const int dr = (c == WHITE) ? 1 : -1;
            if (f > 0 && r + dr >= 0 && r + dr < 8)
                t[c][s] |= 1ULL << ((r + dr) * 8 + f - 1);
            if (f < 7 && r + dr >= 0 && r + dr < 8)
                t[c][s] |= 1ULL << ((r + dr) * 8 + f + 1);
        }
    }
    return t;
}();

inline constexpr std::array<Bitboard, 64> KNIGHT_ATTACKS = [] {
    std::array<Bitboard, 64> t{};
    constexpr int df[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    constexpr int dr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        for (int i = 0; i < 8; ++i) {
            const int nf = f + df[i], nr = r + dr[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                t[s] |= 1ULL << (nr * 8 + nf);
        }
    }
    return t;
}();

inline constexpr std::array<Bitboard, 64> KING_ATTACKS = [] {
    std::array<Bitboard, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) continue;
                const int nf = f + df, nr = r + dr;
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                    t[s] |= 1ULL << (nr * 8 + nf);
            }
    }
    return t;
}();

// ---------------------------------------------------------------------------
// Slider rays: RAYS[dir][s] = all squares strictly beyond s along dir.
// Index: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7.
// ---------------------------------------------------------------------------

enum Dir : int { DIR_N = 0, DIR_NE = 1, DIR_E = 2, DIR_SE = 3, DIR_S = 4, DIR_SW = 5, DIR_W = 6, DIR_NW = 7, DIR_NB = 8 };

// Directions whose square index increases along the ray (nearest blocker = lsb).
constexpr bool dir_positive(Dir d) { return d == DIR_N || d == DIR_NE || d == DIR_E || d == DIR_NW; }

inline constexpr std::array<std::array<Bitboard, 64>, DIR_NB> RAYS = [] {
    std::array<std::array<Bitboard, 64>, DIR_NB> t{};
    constexpr int df[DIR_NB] = {0, 1, 1, 1, 0, -1, -1, -1};
    constexpr int dr[DIR_NB] = {1, 1, 0, -1, -1, -1, 0, 1};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        for (int d = 0; d < DIR_NB; ++d) {
            int nf = f + df[d], nr = r + dr[d];
            while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                t[d][s] |= 1ULL << (nr * 8 + nf);
                nf += df[d];
                nr += dr[d];
            }
        }
    }
    return t;
}();

// Human-readable square name, e.g. make_square(3, 0) -> "d1".
inline std::string square_name(Square s) {
    std::string n;
    n += static_cast<char>('a' + file_of(s));
    n += static_cast<char>('1' + rank_of(s));
    return n;
}

inline Square square_from_name(std::string_view sv) {
    if (sv.size() < 2) return static_cast<Square>(-1);
    const int f = sv[0] - 'a', r = sv[1] - '1';
    if (f < 0 || f > 7 || r < 0 || r > 7) return static_cast<Square>(-1);
    return make_square(f, r);
}

}  // namespace lo
