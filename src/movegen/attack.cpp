#include "movegen/attack.hpp"

#if defined(__x86_64__) || defined(__i386__)
#define LO_X86 1
#include <immintrin.h>
#else
#define LO_X86 0
#endif

#include <cstdlib>
#include <cstring>

namespace lo::att {

namespace {

#if LO_X86
inline bool cpu_has_bmi2() { return __builtin_cpu_supports("bmi2") != 0; }
#else
inline bool cpu_has_bmi2() { return false; }
#endif

bool g_has_bmi2 = false;

// Magic multipliers, one per square, discovered at init time from a
// fixed-seed xorshift64* so tables are deterministic across runs.
std::uint64_t g_rook_magic[64];
std::uint64_t g_bishop_magic[64];

std::uint64_t rng_state = 0x2545F4914F6CDD1DULL;
inline std::uint64_t rng_next() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}
inline std::uint64_t sparse_random() { return rng_next() & rng_next() & rng_next(); }

// Enumerate every subset of mask (carry-rippler), call fn on each.
template <typename Fn>
inline void for_each_subset(Bitboard mask, Fn&& fn) {
    Bitboard occ = 0;
    do {
        fn(occ);
        occ = (occ - mask) & mask;
    } while (occ);
}

void init_masks() {
    for (int s = 0; s < 64; ++s) {
        const Square sq = static_cast<Square>(s);
        rook_mask_table[s] =
            (RAYS[DIR_N][sq] & ~RANK_8) | (RAYS[DIR_E][sq] & ~FILE_H) |
            (RAYS[DIR_S][sq] & ~RANK_1) | (RAYS[DIR_W][sq] & ~FILE_A);
        bishop_mask_table[s] =
            (RAYS[DIR_NE][sq] & ~(RANK_8 | FILE_H)) | (RAYS[DIR_SE][sq] & ~(RANK_1 | FILE_H)) |
            (RAYS[DIR_SW][sq] & ~(RANK_1 | FILE_A)) | (RAYS[DIR_NW][sq] & ~(RANK_8 | FILE_A));
    }
}

#if LO_X86
__attribute__((target("bmi2"))) inline std::uint64_t pext64(std::uint64_t v, std::uint64_t m) {
    return _pext_u64(v, m);
}

__attribute__((target("bmi2"))) Bitboard rook_pext_fn(Square s, Bitboard occ) {
    return rook_table[static_cast<std::size_t>(s) * ROOK_STRIDE +
                      pext64(occ, rook_mask_table[s])];
}
__attribute__((target("bmi2"))) Bitboard bishop_pext_fn(Square s, Bitboard occ) {
    return bishop_table[static_cast<std::size_t>(s) * BISHOP_STRIDE +
                        pext64(occ, bishop_mask_table[s])];
}

void init_pext() {
    for (int s = 0; s < 64; ++s) {
        const Square sq = static_cast<Square>(s);
        for_each_subset(rook_mask_table[s], [&](Bitboard occ) {
            rook_table[static_cast<std::size_t>(s) * ROOK_STRIDE +
                       pext64(occ, rook_mask_table[s])] = classical_rook(sq, occ);
        });
        for_each_subset(bishop_mask_table[s], [&](Bitboard occ) {
            bishop_table[static_cast<std::size_t>(s) * BISHOP_STRIDE +
                         pext64(occ, bishop_mask_table[s])] = classical_bishop(sq, occ);
        });
    }
    rook_attacks = &rook_pext_fn;
    bishop_attacks = &bishop_pext_fn;
}
#endif

// ---------------------------------------------------------------------------
// Magic fallback
// ---------------------------------------------------------------------------

inline std::size_t magic_shift(const Bitboard* mask_table, int s) {
    return 64 - popcount(mask_table[s]);
}

Bitboard magic_rook_fn(Square s, Bitboard occ) {
    const std::size_t idx =
        ((occ & rook_mask_table[s]) * g_rook_magic[s]) >> magic_shift(rook_mask_table, s);
    return rook_table[static_cast<std::size_t>(s) * ROOK_STRIDE + idx];
}
Bitboard magic_bishop_fn(Square s, Bitboard occ) {
    const std::size_t idx =
        ((occ & bishop_mask_table[s]) * g_bishop_magic[s]) >> magic_shift(bishop_mask_table, s);
    return bishop_table[static_cast<std::size_t>(s) * BISHOP_STRIDE + idx];
}

// Search a unique-mapping multiplier for one square. Scratch buffers are
// file-scope: init runs single-threaded before any search activity.
std::uint32_t g_stamp[ROOK_STRIDE];
Bitboard g_seen[ROOK_STRIDE];
Bitboard g_subset_occ[ROOK_STRIDE];
Bitboard g_subset_atk[ROOK_STRIDE];

bool find_magic(int s, bool rook_like) {
    const Bitboard mask = rook_like ? rook_mask_table[s] : bishop_mask_table[s];
    const std::size_t stride = rook_like ? ROOK_STRIDE : BISHOP_STRIDE;
    const std::size_t bits = popcount(mask);
    const std::size_t shift = 64 - bits;

    std::size_t n = 0;
    for_each_subset(mask, [&](Bitboard occ) {
        g_subset_occ[n] = occ;
        g_subset_atk[n] = rook_like ? classical_rook(static_cast<Square>(s), occ)
                                    : classical_bishop(static_cast<Square>(s), occ);
        ++n;
    });

    std::memset(g_stamp, 0, sizeof(std::uint32_t) * stride);
    std::uint32_t epoch = 0;
    Bitboard* table = rook_like ? rook_table : bishop_table;
    const std::size_t base = static_cast<std::size_t>(s) * stride;

    for (int trial = 0; trial < 1000000; ++trial) {
        const std::uint64_t magic = sparse_random();
        if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6)
            continue;  // cheap density heuristic: keeps candidate search short
        ++epoch;
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx = static_cast<std::size_t>((g_subset_occ[i] * magic) >> shift);
            if (g_stamp[idx] != epoch) {
                g_stamp[idx] = epoch;
                g_seen[idx] = g_subset_atk[i];
            } else if (g_seen[idx] != g_subset_atk[i]) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx = static_cast<std::size_t>((g_subset_occ[i] * magic) >> shift);
            table[base + idx] = g_subset_atk[i];
        }
        if (rook_like)
            g_rook_magic[s] = magic;
        else
            g_bishop_magic[s] = magic;
        return true;
    }
    return false;
}

void init_magic() {
    for (int s = 0; s < 64; ++s) {
        if (!find_magic(s, true)) return;  // fall back to classical pointers
        if (!find_magic(s, false)) return;
    }
    rook_attacks = &magic_rook_fn;
    bishop_attacks = &magic_bishop_fn;
}

}  // namespace

bool has_bmi2() { return g_has_bmi2; }

void init() {
    static bool done = false;
    if (done) return;
    done = true;

    init_masks();

    // LO_FORCE_MAGIC=1 exercises the fallback path on BMI2 hardware so both
    // implementations stay verified.
    const bool force_magic = std::getenv("LO_FORCE_MAGIC") != nullptr;

#if LO_X86
    if (!force_magic && cpu_has_bmi2()) {
        g_has_bmi2 = true;
        init_pext();
        return;
    }
#endif
    (void)force_magic;
    init_magic();  // on failure the pointers keep the classical implementation
}

}  // namespace lo::att
