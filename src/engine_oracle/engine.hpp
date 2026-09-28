#pragma once

// Engine A. Milestone-M0 form: material + piece-square argmax gated by the
// symbolic invariants. M1 adds the neural path: a loaded network replaces
// evaluation with policy scores while the gates stay identical.

#include "nn/net.hpp"
#include "nn/netq.hpp"
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

// Own remaining time in milliseconds (INT64_MAX when unknown). Below
// kLowTimeMs the neural path defers to the instant PST path: the reference
// inference costs ~0.3-0.7 s and flagging wastes a full point.
Move bestmove_net(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                  const nn::Net& net, std::int64_t own_time_ms);

// Same gates, INT8 quantized inference.
Move bestmove_net_q(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                    const nn::NetQ& net, std::int64_t own_time_ms);

}  // namespace lo::engine
