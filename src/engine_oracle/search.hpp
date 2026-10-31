#pragma once

// Division-3 engine: iterative-deepening alpha-beta over the INT8 net's
// value head. Same identity (searchless nets serve 0-ply/1-ply divisions);
// this file adds the searched division for CCRL-style events.

#include "engine_oracle/engine.hpp"

#include <cstdint>

namespace lo::engine {

// ABNodes: hard node cap per decision. ABTimeMs: soft wall-clock cap.
Move bestmove_ab(const PositionState& p, const std::vector<std::uint64_t>& key_history,
                 const nn::NetQ& net, std::int64_t own_time_ms,
                 std::uint64_t ab_nodes, std::int64_t ab_time_ms);

}  // namespace lo::engine
