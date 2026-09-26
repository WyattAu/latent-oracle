#pragma once

// Engine A. Milestone-M0 form: material + piece-square argmax gated by the
// symbolic invariants. M1 adds the neural path: a loaded network replaces
// evaluation with policy scores while the gates stay identical.

#include "nn/net.hpp"
#include "position.hpp"

#include <cstdint>
#include <vector>

namespace lo::engine {

// White-relative static evaluation in centipawns (M0 baseline).
int evaluate(const PositionState& p);

// Deterministic best move (PST eval), or MOVE_NONE if checkmate/stalemate.
Move bestmove(const PositionState& p, const std::vector<std::uint64_t>& key_history);

// Neural path: policy scores from `net`, draw gates via its WDL head and the
// same symbolic invariants. Deterministic.
Move bestmove_net(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                  const nn::Net& net);

}  // namespace lo::engine
