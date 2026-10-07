#include "uci/uci.hpp"

#include "engine_oracle/engine.hpp"
#include "movegen/movegen.hpp"
#include "nn/net.hpp"
#include "nn/netq.hpp"
#include "tb/tb.hpp"
#include "uci/spsc_ring.hpp"

#include <atomic>
#include <chrono>
#include <limits>
#include <optional>
#include <cstdio>
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
    char buf[4096];  // long "position ... moves ..." replays
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
    std::vector<lo::Move> game_moves;    // for HiCo history (last 3 plies)
    std::string weights_path;            // empty => PST fallback
    std::optional<lo::nn::Net> net;      // FP32 ("LONW")
    std::optional<lo::nn::NetQ> netq;    // INT8 ("LOQW"); preferred when set
    int recycle = 1;                     // RecyclePasses (looped trunk)
    bool mirror_avg = true;              // MirrorAvg (policy mirror averaging):
                                         // +34.86 +/- 19.63 Elo over 400 games
                                         // (RESULTS.md, 2026-10-07); costs one extra
                                         // forward pass, still one decision per move
    float loopcd_alpha = 0.0f;           // LoopCD contrastive-decoding strength
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
                s.game_moves.push_back(ml.moves[i]);
                if (s.game_moves.size() > 3) s.game_moves.erase(s.game_moves.begin());
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
    s.game_moves.clear();
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
        std::cout << "option name RecyclePasses type spin default 1 min 1 max 8\n";
        std::cout << "option name MirrorAvg type check default true\n";
        std::cout << "option name LoopCDAlpha type string default 0.0\n";
        std::cout << "uciok\n" << std::flush;
    } else if (cmd == "isready") {
        std::cout << "readyok\n" << std::flush;
    } else if (cmd == "ucinewgame") {
        s.has_root = false;
        s.history.clear();
        s.game_moves.clear();
    } else if (cmd == "setoption") {
        // setoption name WeightsFile value <path>
        auto npos = line.find("name ");
        auto vpos = line.find(" value ");
        if (npos != std::string::npos && vpos != std::string::npos) {
            std::string name = line.substr(npos + 5, vpos - (npos + 5));
            // trim
            while (!name.empty() && name.back() == ' ') name.pop_back();
            if (name == "SyzygyPath") {
                const bool ok = lo::tb::init(line.substr(vpos + 7));
                std::cout << "info string syzygy " << (ok ? "loaded" : "FAILED") << "\n" << std::flush;
            } else if (name == "WeightsFile") {
                s.weights_path = line.substr(vpos + 7);
                // magic-dispatch: "LONW" -> FP32 Net, "LOQW" -> INT8 NetQ
                std::uint32_t magic = 0;
                if (std::FILE* fp = std::fopen(s.weights_path.c_str(), "rb")) {
                    if (std::fread(&magic, 4, 1, fp) != 1) magic = 0;
                    std::fclose(fp);
                }
                s.net.reset();
                s.netq.reset();
                if (magic == 0x57514F4CU) {  // "LOQW" LE
                    s.netq = lo::nn::NetQ::load(s.weights_path);
                } else {
                    s.net = lo::nn::Net::load(s.weights_path);
                    if (s.net) s.net->set_inference(s.recycle, s.loopcd_alpha);
                }
                if (!s.net && !s.netq)
                    std::cout << "info string cannot load weights: " << s.weights_path << "\n" << std::flush;
            } else if (name == "MirrorAvg") {
                const std::string v = line.substr(vpos + 7);
                s.mirror_avg = (v == "true" || v == "1");
            } else if (name == "RecyclePasses") {
                try {
                    s.recycle = std::max(1, std::stoi(line.substr(vpos + 7)));
                    if (s.net) s.net->set_inference(s.recycle, s.loopcd_alpha);
                } catch (...) {}
            } else if (name == "LoopCDAlpha") {
                try {
                    s.loopcd_alpha = std::stof(line.substr(vpos + 7));
                    if (s.net) s.net->set_inference(s.recycle, s.loopcd_alpha);
                } catch (...) {}
            }
        }
    } else if (cmd == "position") {
        set_position(s, line);
    } else if (cmd == "go") {
        g_stop.store(false, std::memory_order_relaxed);
        // Own remaining time from the go parameters (wtime when white, btime
        // when black); INT64_MAX when absent.
        std::int64_t own_time = std::numeric_limits<std::int64_t>::max();
        {
            std::istringstream gs(line);
            std::string tok;
            std::int64_t wtime = -1, btime = -1;
            bool have_clock = false;
            while (gs >> tok) {
                if (tok == "wtime" || tok == "btime") {
                    std::int64_t v = -1;
                    if (gs >> v) {
                        (tok == "wtime" ? wtime : btime) = v;
                        have_clock = true;
                    }
                }
            }
            if (have_clock && s.has_root)
                own_time = (s.root.side_to_move == 0) ? wtime : btime;
            if (own_time < 0) own_time = std::numeric_limits<std::int64_t>::max();
        }
        Move best = MOVE_NONE;
        if (s.has_root) {
            lo::nn::NetHistory hist;
            for (std::size_t i = 0; i < s.game_moves.size() && i < 3; ++i) {
                hist.from[i] = static_cast<std::uint8_t>(move_from(s.game_moves[i]));
                hist.to[i] = static_cast<std::uint8_t>(move_to(s.game_moves[i]));
            }
            hist.n = static_cast<std::uint8_t>(std::min<std::size_t>(3, s.game_moves.size()));
            if (s.netq)
                best = engine::bestmove_net_q(s.root, s.history, *s.netq, own_time, hist,
                                              s.mirror_avg);
            else if (s.net)
                best = engine::bestmove_net(s.root, s.history, *s.net, own_time, hist,
                                            s.mirror_avg);
            else
                best = engine::bestmove(s.root, s.history);
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
