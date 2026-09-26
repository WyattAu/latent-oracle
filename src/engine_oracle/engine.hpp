#pragma once

// Engine A, milestone-M0 form: a material + piece-square argmax over legal
// moves, gated by the symbolic invariants that survive into every later
// version: three-fold repetition, the 75-move automatic draw, and dead-
// position detection. The neural policy replaces `evaluate` in M1/M2; the
// gates stay.

#include "position.hpp"

#include <cstdint>
#include <vector>

namespace lo::engine {

// White-relative static evaluation in centipawns.
int evaluate(const PositionState& p);

// Deterministic best move for the side to move, or MOVE_NONE if checkmate/
// stalemate. `key_history` holds the Zobrist keys of all positions since the
// root of the game (root included), used for repetition detection.
Move bestmove(const PositionState& p, const std::vector<std::uint64_t>& key_history);

}  // namespace lo::engine
