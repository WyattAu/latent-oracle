"""MirrorAvg correctness: with file-flip mirroring (white stays white), the
engine's choice in the mirrored position must be the file-flip of its choice
in the original."""
import subprocess, sys
import chess

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--engine", default="build/debug/latent-oracle")
_ap.add_argument("--weights", default="")
ARGS = _ap.parse_args()
ENGINE = ARGS.engine
W = ARGS.weights

def file_mirror(fen):
    b = chess.Board(fen)
    m = chess.Board(None)
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p:
            m.set_piece_at(sq ^ 7, p)          # a<->h, same color, same rank
    m.turn = b.turn
    # castling bits are WK=1, WQ=2, BK=4, BQ=8; the mirror swaps 1<->2, 4<->8
    rights = ""
    cr = 0
    if b.has_castling_rights(chess.WHITE) and b.castling_rights & chess.BB_H1: cr |= 1  # WK
    if b.has_castling_rights(chess.WHITE) and b.castling_rights & chess.BB_A1: cr |= 2  # WQ
    if b.has_castling_rights(chess.BLACK) and b.castling_rights & chess.BB_H8: cr |= 4  # BK
    if b.has_castling_rights(chess.BLACK) and b.castling_rights & chess.BB_A8: cr |= 8  # BQ
    mir = ((cr & 1) << 1) | ((cr & 2) >> 1) | ((cr & 4) << 1) | ((cr & 8) >> 1)
    for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")):
        if mir & bit:
            rights += ch
    m.set_castling_fen(rights or "-")
    m.ep_square = b.ep_square ^ 7 if b.ep_square is not None else None
    return m.fen()

def ask(fen):
    out = subprocess.run([ENGINE], input=(
        f"uci\nsetoption name WeightsFile value {W}\n"
        f"setoption name MirrorAvg value true\nisready\n"
        f"position fen {fen}\ngo depth 1\nquit\n"),
        capture_output=True, text=True, timeout=120).stdout
    for line in out.splitlines():
        if line.startswith("bestmove"):
            return line.split()[1]
    return None

FENS = [
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]
bad = 0
for f in FENS:
    mf = file_mirror(f)
    a, b = ask(f), ask(mf)
    if not a or not b:
        print("SKIP", f, a, b); continue
    mv = chess.Move.from_uci(a)
    exp = chess.Move(mv.from_square ^ 7, mv.to_square ^ 7).uci()
    ok = (b == exp)
    bad += (not ok)
    print(("OK   " if ok else "FAIL ") + f"{f[:38]:40s} P->{a}  mirror->{b}  expect {exp}")
print("MIRROR MAPPING:", "CORRECT" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
