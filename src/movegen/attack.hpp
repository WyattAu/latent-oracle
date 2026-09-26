#pragma once

// Slider attack machinery.
//
// Three implementations exist:
//   1. Classical ray attacks (always available, used to bootstrap both others).
//   2. Magic bitboards (fallback when BMI2 is absent).
//   3. PEXT bitboards (primary when BMI2 is present).
//
// att::init() performs runtime CPU detection and fills exactly the tables the
// selected implementation needs, then repoints the dispatch function pointers.
// Before init() the pointers resolve to the classical implementation, so the
// engine is always correct regardless of call order.

#include "types.hpp"

namespace lo {

// Classical ray-walk attacks: always correct, no tables required.
inline Bitboard classical_slider(Square s, Bitboard occ, std::initializer_list<Dir> dirs) {
    Bitboard atk = 0;
    for (Dir d : dirs) {
        Bitboard ray = RAYS[d][s];
        const Bitboard blockers = ray & occ;
        if (blockers) {
            const Square first = static_cast<Square>(
                dir_positive(d) ? lsb(blockers) : msb(blockers));
            atk |= ray ^ RAYS[d][first];
        } else {
            atk |= ray;
        }
    }
    return atk;
}

inline Bitboard classical_rook(Square s, Bitboard occ) {
    return classical_slider(s, occ, {DIR_N, DIR_E, DIR_S, DIR_W});
}
inline Bitboard classical_bishop(Square s, Bitboard occ) {
    return classical_slider(s, occ, {DIR_NE, DIR_SE, DIR_SW, DIR_NW});
}

namespace att {

// Dispatch points. Initialized to the classical implementation; init() may
// repoint to PEXT or magic. Call through these, never through the classical
// functions directly, outside of table bootstrap.
inline Bitboard (*rook_attacks)(Square, Bitboard) = &classical_rook;
inline Bitboard (*bishop_attacks)(Square, Bitboard) = &classical_bishop;

// Table strides: rook masks have at most 12 bits (4096 subsets), bishop masks
// at most 9 bits (512 subsets). Fixed stride trades ~2.3 MB of lazy BSS for
// division-free indexing.
constexpr std::size_t ROOK_STRIDE = 4096;
constexpr std::size_t BISHOP_STRIDE = 512;

inline Bitboard rook_mask_table[64];
inline Bitboard bishop_mask_table[64];
inline Bitboard rook_table[64 * ROOK_STRIDE];
inline Bitboard bishop_table[64 * BISHOP_STRIDE];

// Detect CPU features, build masks + the tables for the selected
// implementation, and repoint the dispatch pointers. Deterministic: magic
// candidates are drawn from a fixed-seed xorshift, so tables are identical
// across runs.
void init();

bool has_bmi2();

}  // namespace att

inline Bitboard rook_attacks(Square s, Bitboard occ) { return att::rook_attacks(s, occ); }
inline Bitboard bishop_attacks(Square s, Bitboard occ) { return att::bishop_attacks(s, occ); }

}  // namespace lo
