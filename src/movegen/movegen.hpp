#pragma once

// Move generation. Legal generation is pseudo-legal + copy-make verification:
// every candidate is applied and rejected if the mover's king is attacked.
// Correctness first; pin-aware filtering is an M2 optimization behind the
// same interface.

#include "position.hpp"

namespace lo {

// All pseudo-legal moves (castling generation pre-checks the king's path so
// only the landing square needs post-verification).
void generate_pseudo(const PositionState& p, MoveList& out);

// Strictly legal moves.
void generate_legal(const PositionState& p, MoveList& out);

}  // namespace lo
