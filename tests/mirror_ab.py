#!/usr/bin/env python3
"""Paired A/B for MirrorAvg at the position level.

A game match would need hundreds of games to resolve a few Elo, which the
shared box cannot afford. This measures the same thing with paired samples:
take real positions, ask the engine for its move with MirrorAvg off and on,
and for the positions where the choice differs, evaluate both moves with
Stockfish. Each position is its own control, so the comparison has far lower
variance than a game-level SPRT.

Reports:
  - how often averaging changes the move
  - the paired eval difference (net-POV centipawns) over changed positions
  - a bootstrap interval on that mean

Usage:
  python3 tests/mirror_ab.py --pgn <file.pgn> --weights <blob> \
      --engine build/debug/latent-oracle [--positions 1500] [--depth 10]
"""
from __future__ import annotations

import argparse
import random
import statistics
import subprocess
import sys

import chess
import chess.pgn


def positions_from_pgn(path: str, want: int, seed: int, skip_ply: int = 8) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    with open(path) as fh:
        while len(out) < want:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            n = 0
            for move in game.mainline_moves():
                if n >= skip_ply:
                    out.append(board.fen())
                    if len(out) >= want:
                        break
                board.push(move)
                n += 1
    rng.shuffle(out)
    return out[:want]


class Engine:
    """One long-lived engine process fed many positions."""

    def __init__(self, path: str, weights: str, mirror: bool):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send(f"setoption name WeightsFile value {weights}")
        self._send(f"setoption name MirrorAvg value {'true' if mirror else 'false'}")
        self._send("uci")
        self._wait("uciok")
        self._send("isready")
        self._wait("readyok")

    def _send(self, cmd: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _wait(self, token: str) -> None:
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line or token in line:
                return

    def bestmove(self, fen: str) -> str | None:
        self._send(f"position fen {fen}")
        self._send("go depth 1")
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line:
                return None
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None

    def close(self) -> None:
        try:
            self.p.kill()
        except Exception:  # noqa: BLE001
            pass


class SF:
    def __init__(self, path: str, depth: int, threads: int = 2):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.depth = depth
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value 32")
        self._send("uci")
        self._wait("uciok")
        self._send("isready")
        self._wait("readyok")

    def _send(self, cmd: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _wait(self, token: str) -> None:
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line or token in line:
                return

    def close(self) -> None:
        try:
            self.p.kill()
        except Exception:  # noqa: BLE001
            pass

    def cp_side_to_move(self, board: chess.Board) -> int | None:
        self._send(f"position fen {board.fen()}")
        self._send(f"go depth {self.depth}")
        best = None
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line:
                return best
            if line.startswith("bestmove"):
                return best
            if line.startswith("info") and " score " in line:
                toks = line.split()
                try:
                    i = toks.index("score")
                    kind, val = toks[i + 1], int(toks[i + 2])
                except (ValueError, IndexError):
                    continue
                if kind == "cp":
                    best = val
                elif kind == "mate":
                    best = 30000 if val > 0 else -30000


def net_pov_after(board: chess.Board, uci: str, mover: chess.Color, sf: SF) -> int | None:
    """Evaluation after playing uci, expressed for `mover`."""
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if mv not in board.legal_moves:
        return None
    board.push(mv)
    cp = sf.cp_side_to_move(board)
    board.pop()
    if cp is None or abs(cp) >= 30000:
        return None
    return cp if mover == chess.WHITE else -cp


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(statistics.fmean(rng.choice(values) for _ in range(n)))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--engine", default="build/debug/latent-oracle")
    ap.add_argument("--sf", default="/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    fens = positions_from_pgn(args.pgn, args.positions, args.seed)
    print(f"positions: {len(fens)}")

    off = Engine(args.engine, args.weights, mirror=False)
    on = Engine(args.engine, args.weights, mirror=True)
    sf = SF(args.sf, args.depth)
    changed: list[float] = []
    n_same = 0
    try:
        for i, fen in enumerate(fens):
            if i and i % 50 == 0:
                print(f"  {i}/{len(fens)} ({n_same} same, {len(changed)} changed)",
                      flush=True)
            a = off.bestmove(fen)
            b = on.bestmove(fen)
            if not a or not b:
                continue
            if a == b:
                n_same += 1
                continue
            board = chess.Board(fen)
            mover = board.turn
            va = net_pov_after(board, a, mover, sf)
            vb = net_pov_after(board, b, mover, sf)
            if va is None or vb is None:
                continue
            changed.append(vb - va)
    finally:
        off.close()
        on.close()
        sf.close()

    total = n_same + len(changed) + (len(fens) - n_same - len(changed))
    print(f"- identical move: {n_same}/{total} ({100*n_same/max(1,total):.1f}%)")
    print(f"- evaluated changed positions: {len(changed)}")
    if not changed:
        print("MirrorAvg: no measurable difference")
        return 0
    mean = statistics.fmean(changed)
    lo, hi = bootstrap_ci(changed)
    better = sum(1 for v in changed if v > 0)
    worse = sum(1 for v in changed if v < 0)
    print(f"- paired eval difference (on - off): mean {mean:+.1f}cp, "
          f"95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"- better {better} / worse {worse}")
    verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "inconclusive")
    print(f"MIRRORAVG VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
