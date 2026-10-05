#!/usr/bin/env python3
"""CI parity gate: tiny random nets through the full export -> engine ->
compare loop for BOTH inference paths. Fails the build if the engine's FP32
or INT8 output drifts from the Python reference.

Usage (from repo root, after building the engine):
    python3 tests/ci_parity.py --engine build/debug/latent-oracle
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import chess
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO.parent / "latent-oracle-data" / "trainer"))

FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kk - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"]


def fen_codes(fen: str):
    board = chess.Board(fen)
    codes = np.zeros(64, dtype=np.int64)
    base = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4,
            chess.QUEEN: 5, chess.KING: 6}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            codes[sq] = base[p.piece_type] + (0 if p.color == chess.WHITE else 8)
    return codes, 0 if board.turn == chess.WHITE else 1


def check(engine: str, blob: str, cmd: str, wdl_py: dict, tol: float) -> bool:
    ok = True
    for fen in FENS:
        out = subprocess.run([engine, cmd, "--weights", blob, "--fen"] + fen.split(),
                             capture_output=True, text=True).stdout.strip().split("\n")
        cpp_wdl = [float(x) for x in out[0].split()[1:]]
        err = max(abs(a - b) for a, b in zip(cpp_wdl, wdl_py[fen]))
        status = "ok" if err < tol else f"DIVERGES ({err:.2e})"
        if err >= tol:
            ok = False
        print(f"  {fen[:30]:32s} wdl_err={err:.2e} {status}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="build/debug/latent-oracle")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO.parent / "latent-oracle-data" / "trainer"))
    from model import ChessNet  # noqa: E402
    from loqw_sim import Loqw, forward  # noqa: E402

    torch.manual_seed(11)
    tmp = Path("/tmp")
    m = ChessNet(d=32, layers=2, heads=4, dff=32, dpol=16)

    # ---- FP32 path
    m.export_blob(str(tmp / "ci_v1.bin"))
    torch.save(m.state_dict(), str(tmp / "ci_v1.pt"))
    ref = {}
    import export_parity as ep  # reuse the FEN encoder if importable
    for fen in FENS:
        codes, side = fen_codes(fen)
        with torch.no_grad():
            s, p, w = m(torch.from_numpy(codes).unsqueeze(0), torch.tensor([side]))
        ref[fen] = [float(v) for v in torch.softmax(w[0], -1)]
    print("FP32 parity:")
    ok32 = check(args.engine, str(tmp / "ci_v1.bin"), "nnpar", ref, 1e-4)

    # ---- INT8 path
    m.export_blob(str(tmp / "ci_q_v1.bin"))  # not used; export_int8 writes LOQW
    sys.path.insert(0, str(REPO.parent / "latent-oracle-data" / "trainer"))
    import importlib
    import export_int8 as ei
    importlib.reload(ei)
    sys.argv = ["export_int8.py", "--net", str(tmp / "ci_v1.pt"), "--d", "32",
                "--layers", "2", "--heads", "4", "--dff", "32", "--dpol", "16",
                "--calib", "64", "--out", str(tmp / "ci_q.bin")]
    ei.main()
    mq = Loqw(str(tmp / "ci_q.bin"))
    mq.parse()
    refq = {}
    for fen in FENS:
        codes, side = fen_codes(fen)
        with torch.no_grad():
            scores, promo, wdl = forward(mq, torch.from_numpy(codes).unsqueeze(0), side)
        refq[fen] = [float(v) for v in torch.softmax(wdl, -1)]
    print("INT8 parity:")
    okq = check(args.engine, str(tmp / "ci_q.bin"), "nnqpar", refq, 5e-3)

    print("PARITY GATE:", "PASS" if (ok32 and okq) else "FAIL")
    sys.exit(0 if (ok32 and okq) else 1)


if __name__ == "__main__":
    main()
