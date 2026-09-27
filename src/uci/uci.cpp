#include "uci/uci.hpp"

#include "engine_oracle/engine.hpp"
#include "movegen/movegen.hpp"
#include "nn/net.hpp"
#include "uci/spsc_ring.hpp"

#include <atomic>
#include <chrono>
#include <optional>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace lo::uci {

namespace {

constexpr std::size_t kRingCapacity = 256;

struct Line {
    std::size_t len = 0;
    char buf[1024];  // long "position ... moves ..." replays
};

SpscRing<Line, kRingCapacity> g_ring;
std::atomic<bool> g_quit{false};
std::atomic<bool> g_stop{false};
std::atomic<bool> g_reader_done{false};

void reader_loop() {
    std::string line;
    while (!g_quit.load(std::memory_order_relaxed)) {
        if (!std::getline(std::cin, line)) break;
        Line l;
        l.len = line.size() < sizeof(l.buf) - 1 ? line.size() : sizeof(l.buf) - 1;
        std::memcpy(l.buf, line.data(), l.len);
        while (!g_ring.push(l)) {
            if (g_quit.load(std::memory_order_relaxed)) {
                g_reader_done.store(true, std::memory_order_release);
                return;
            }
            std::this_thread::yield();
        }
        if (line == "quit") break;
    }
    g_reader_done.store(true, std::memory_order_release);
}

struct Session {
    PositionState root{};
    bool has_root = false;
    std::vector<std::uint64_t> history;  // Zobrist keys, root position included
    std::string weights_path;            // empty => PST fallback
    std::optional<lo::nn::Net> net;
};

// Applies a sequence of UCI move tokens to the session root. A token that
// matches no legal move aborts the replay (never desync from the GUI).
void apply_moves(Session& s, std::istringstream& iss) {
    std::string mstr;
    while (iss >> mstr) {
        MoveList ml;
        generate_legal(s.root, ml);
        bool found = false;
        for (std::uint32_t i = 0; i < ml.count; ++i) {
            if (move_to_uci(ml.moves[i]) == mstr) {
                s.root = apply(s.root, ml.moves[i]);
                s.history.push_back(s.root.zobrist);
                found = true;
                break;
            }
        }
        if (!found) return;
    }
}

void set_position(Session& s, const std::string& line) {
    std::istringstream iss(line);
    std::string cmd, kind;
    iss >> cmd >> kind;

    std::string fen;
    // UCI grammar: position [fen <fen> | startpos] [moves <move>...].
    //
    // The "moves" keyword belongs to the grammar, and each branch must leave
    // the stream positioned at the first move token. The FEN branch consumes
    // it inside its field loop (breaking on it); the startpos branch has no
    // fields to skip, so the first token after "startpos" is already a move.
    // A previous revision checked for the keyword AGAIN here, which consumed
    // the first move of `position fen <fen> moves ...` and silently ignored
    // the whole command — the engine then answered a stale root mid-game.
    if (kind == "startpos") {
        fen = std::string(STARTPOS_FEN);
    } else if (kind == "fen") {
        std::string t;
        while (iss >> t) {
            if (t == "moves") break;
            fen += t;
            fen += ' ';
        }
    } else {
        return;
    }

    const auto parsed = parse_fen(fen);
    if (!parsed) return;
    s.root = *parsed;
    s.has_root = true;
    s.history.clear();
    s.history.push_back(s.root.zobrist);
    apply_moves(s, iss);
}

void handle_line(Session& s, const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    if (!(iss >> cmd)) return;

    if (cmd == "uci") {
        std::cout << "id name latent-oracle 0.1.0\n";
        std::cout << "id author Wyatt Au\n";
        std::cout << "option name WeightsFile type string default\n";
        std::cout << "uciok\n" << std::flush;
    } else if (cmd == "isready") {
        std::cout << "readyok\n" << std::flush;
    } else if (cmd == "ucinewgame") {
        s.has_root = false;
        s.history.clear();
    } else if (cmd == "setoption") {
        // setoption name WeightsFile value <path>
        auto npos = line.find("name ");
        auto vpos = line.find(" value ");
        if (npos != std::string::npos && vpos != std::string::npos) {
            std::string name = line.substr(npos + 5, vpos - (npos + 5));
            // trim
            while (!name.empty() && name.back() == ' ') name.pop_back();
            if (name == "WeightsFile") {
                s.weights_path = line.substr(vpos + 7);
                s.net = lo::nn::Net::load(s.weights_path);
                if (!s.net)
                    std::cout << "info string cannot load weights: " << s.weights_path << "\n" << std::flush;
            }
        }
    } else if (cmd == "position") {
        set_position(s, line);
    } else if (cmd == "go") {
        g_stop.store(false, std::memory_order_relaxed);
        Move best = MOVE_NONE;
        if (s.has_root) {
            best = s.net ? engine::bestmove_net(s.root, s.history, *s.net)
                         : engine::bestmove(s.root, s.history);
        }
        std::cout << "bestmove " << (best == MOVE_NONE ? std::string("0000") : move_to_uci(best))
                  << "\n"
                  << std::flush;
    } else if (cmd == "stop") {
        g_stop.store(true, std::memory_order_relaxed);
    } else if (cmd == "quit") {
        g_quit.store(true, std::memory_order_relaxed);
    }
    // setoption, ponderhit, register, debug: accepted and ignored in M0.
}

}  // namespace

int run() {
    Session session;
    std::thread reader(reader_loop);

    Line line;
    while (true) {
        if (g_ring.pop(line)) {
            handle_line(session, std::string(line.buf, line.len));
            if (g_quit.load(std::memory_order_relaxed)) break;
        } else if (g_reader_done.load(std::memory_order_acquire)) {
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    reader.join();
    return 0;
}

}  // namespace lo::uci
