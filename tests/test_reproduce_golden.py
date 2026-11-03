"""Golden four-cell reproduction from the SHIPPED model artifacts (no training).

Seam under test: reproduce.evaluate(models_dir, caches_dir) -> {cell: (mIoU, BDE, acc75, rows)}.
Expected values are the paper's Table 1 (reproduce.EXPECT) and the tracked per-image boxes in
inference/results/boxes_*.csv — an independent source of truth, not recomputed by the code.

Usage: python tests/test_reproduce_golden.py [--caches DIR]   (CPU, ~25 s; needs the cache archive)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=str(ROOT / "caches"))
    a = ap.parse_args()
    R = _load("reproduce", ROOT / "inference" / "reproduce.py")
    cells = R.evaluate(ROOT / "inference" / "models", Path(a.caches))
    bad = 0
    for name, (miou, bde, acc, rows) in cells.items():
        e_iou, e_bde, e_acc = R.EXPECT[name]
        ok = abs(miou - e_iou) <= R.TOL and abs(bde - e_bde) <= R.TOL and acc == e_acc
        ship = pd.read_csv(ROOT / "inference" / "results" / f"boxes_{name}.csv")
        got = pd.DataFrame(rows)
        m = ship.merge(got, on="id", suffixes=("_ship", "_got"), validate="one_to_one")
        box_same = int((m[["x1_ship", "y1_ship", "x2_ship", "y2_ship"]].values
                        == m[["x1_got", "y1_got", "x2_got", "y2_got"]].values).all(1).sum())
        d_iou = float((m.iou_ship - m.iou_got).abs().max())
        d_bde = float((m.bde_ship - m.bde_got).abs().max())
        rows_ok = len(m) == len(ship) == len(got) and box_same == len(m) and d_iou < 1e-9 and d_bde < 1e-9
        bad += (not ok) + (not rows_ok)
        print(f"{name:10s} mIoU {miou:.4f} BDE {bde:.4f} Acc {acc:3d}  cell {'PASS' if ok else 'FAIL'} | "
              f"boxes identical {box_same}/{len(m)}  max|dIoU| {d_iou:.1e}  max|dBDE| {d_bde:.1e}  {'PASS' if rows_ok else 'FAIL'}")
    print("PASS: golden reproduction from shipped artifacts" if not bad else f"FAIL: {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
