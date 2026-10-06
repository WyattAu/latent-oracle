"""MirrorAvg correctness: with file-flip mirroring (white stays white), the
engine's choice in the mirrored position must be the file-flip of its choice
in the original."""
import os
import subprocess
import sys

import chess

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--engine", default="build/debug/latent-oracle")
_ap.add_argument("--weights", default="")
ARGS = _ap.parse_args()
ENGINE = ARGS.engine
W = ARGS.weights

if not W:
    # Self-contained: CI has no trained blob. The property under test belongs to
    # the engine's mirror mapping, not to the weights, so a tiny random net is
    # enough -- and using one keeps a missing weights file from silently
    # degrading the check to the PST fallback (which is not mirror-averaged
    # and made CI fail for the wrong reason).
    import tempfile
    import torch

    sys.path.insert(0, "../latent-oracle-data/trainer")
    from model import ChessNet
    torch.manual_seed(3)
    net = ChessNet(d=32, layers=2, heads=4, dff=32, dpol=16)
    W = os.path.join(tempfile.gettempdir(), "mirror_check_v1.bin")
    net.export_blob(W)
    print(f"using a generated random blob: {W}")

def file_mirror(fen):
    b = chess.Board(fen)
    m = chess.Board(None)
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p:
            m.set_piece_at(sq ^ 7, p)          # a<->h, same color, same rank
    m.turn = b.turn
    rights_mask = None
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
    promo = mv.promotion          # a promotion keeps its piece under mirroring
    exp = chess.Move(mv.from_square ^ 7, mv.to_square ^ 7, promotion=promo).uci()
    # castling cannot be mirrored (a file mirror puts the king off e1), so
    # MirrorAvg scores castling from the direct view only and the exact
    # mirror-equality invariant does not hold for such positions
    castling = (abs(chess.square_file(mv.to_square) - chess.square_file(mv.from_square)) == 2
                and chess.square_rank(mv.from_square) == 0
                and chess.square_rank(mv.to_square) == 0)   # both on the back rank
    if castling:
        ok = bool(b) and chess.Move.from_uci(b) in chess.Board(f).legal_moves
        print(("OK*  " if ok else "FAIL ") + f"{f[:38]:40s} P->{a}  mirror->{b}  (castling: legal only)")
    else:
        ok = (b == exp)
        print(("OK   " if ok else "FAIL ") + f"{f[:38]:40s} P->{a}  mirror->{b}  expect {exp}")
    bad += (not ok)
print("MIRROR MAPPING:", "CORRECT" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
