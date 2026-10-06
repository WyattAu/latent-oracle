#!/usr/bin/env python3
"""Differential movegen fuzz: python-chess vs the engine's perft.

perft(1) is exactly the legal-move count, so any disagreement is a movegen
bug. Positions come from seeded random playouts (reproducible, so this is
safe in CI) biased toward castling, en-passant and promotion states -- the
three places this engine has actually had bugs (an inverted ep-capturability
test was found this way and is recorded in docs/RESULTS.md).

Usage:
    python3 tests/fuzz_movegen.py --engine build/debug/latent-oracle \
        [--positions 400] [--deep 25] [--seed 1234]
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys

import chess


def engine_perft(engine: str, fen: str, depth: int) -> int:
    out = subprocess.run([engine, "perft", str(depth), "--fen", fen],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"engine failed on {fen!r}: {out.stderr.strip()[:200]}")
    # the engine prints "nodes: N"; stockfish-style builds print
    # "Nodes searched: N", so accept both
    for line in reversed(out.stdout.strip().splitlines()):
        low = line.lower()
        if low.startswith("nodes:") or low.startswith("nodes searched"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"no node count in engine output: {out.stdout[-200:]!r}")


def random_positions(rng: random.Random, count: int):
    """Yield FENs from random playouts, biased to interesting states."""
    seen = set()
    while len(seen) < count:
        board = chess.Board()
        for _ in range(rng.randint(4, 120)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            # bias: prefer captures/promotions/castle/ep sometimes to reach
            # the tricky states quickly
            interesting = [m for m in moves
                           if board.is_capture(m) or m.promotion
                           or board.is_castling(m)]
            pool = interesting if (interesting and rng.random() < 0.35) else moves
            board.push(rng.choice(pool))
            fen = board.fen()
            interesting_state = (any(board.is_castling(m) for m in moves)
                                 or board.ep_square is not None
                                 or any(m.promotion for m in moves))
            if interesting_state and fen not in seen:
                seen.add(fen)
                break
    return list(seen)[:count]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="build/debug/latent-oracle")
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--deep", type=int, default=25,
                    help="how many of them to check at perft 3")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fens = random_positions(rng, args.positions)
    print(f"fuzzing {len(fens)} random positions (seed {args.seed})")

    bad = []
    for i, fen in enumerate(fens):
        want = chess.Board(fen).legal_moves.count()
        got = engine_perft(args.engine, fen, 1)
        if want != got:
            bad.append((fen, want, got))
            print(f"  MISMATCH perft(1) {fen}\n    python-chess {want} != engine {got}")
            if len(bad) >= 5:
                break
        if i < args.deep:
            want3 = _perft_py(fen, 3)
            got3 = engine_perft(args.engine, fen, 3)
            if want3 != got3:
                bad.append((fen, want3, got3))
                print(f"  MISMATCH perft(3) {fen}\n    python-chess {want3} != engine {got3}")
                if len(bad) >= 5:
                    break

    if bad:
        print(f"FUZZ FAILED: {len(bad)} mismatch(es)")
        return 1
    print(f"MOVEGEN FUZZ PASSED ({len(fens)} positions, {args.deep} at depth 3)")
    return 0


def _perft_py(fen: str, depth: int) -> int:
    if depth == 0:
        return 1
    board = chess.Board(fen)
    if depth == 1 or board.is_game_over():
        return board.legal_moves.count()
    return sum(_perft_py(_child_fen(board, m), depth - 1) for m in board.legal_moves)


def _child_fen(board: chess.Board, move: chess.Move) -> str:
    board.push(move)
    fen = board.fen()
    board.pop()
    return fen


if __name__ == "__main__":
    sys.exit(main())
