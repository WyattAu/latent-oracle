#pragma once

// Syzygy tablebase root probing (Fathom, MIT — see PROVENANCE.md).
//
// Policy: when material is <= 5 men and no castling rights remain, the root
// is probed and the tablebase move overrides the policy when the probe
// succeeds. DTZ values respect the 50-move clock, so this converts won
// endgames provably — the exact blind spot of a searchless policy.

#include "position.hpp"

#include <optional>
#include <string>

namespace lo::tb {

// Loads the tablebase directory. Safe to call repeatedly.
bool init(const std::string& path);

bool available();

// DTZ/WDL-optimal move, or nullopt when the position is out of TB scope
// (>5 men, castling rights, probe failure) or terminal.
std::optional<Move> probe_root(const PositionState& pos);

}  // namespace lo::tb
