#pragma once

namespace lo::uci {

// Runs the UCI loop until "quit" or EOF. Assumes att::init() has completed.
int run();

}  // namespace lo::uci
