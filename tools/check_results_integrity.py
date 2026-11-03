#!/usr/bin/env python3
"""Verify the four shipped per-image result cells against their pinned metrics."""

import argparse
import math
import csv
import json
import re
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "inference" / "results"
EXPECT = {
    "cond_full": (345, 0.765970, 0.055899, 218),
    "cond_fast": (345, 0.759273, 0.058407, 210),
    "ff_full": (500, 0.856323, 0.032806, 468),
    "ff_fast": (500, 0.834630, 0.041183, 424),
}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")


def check_cell(cell, expected):
    count, expected_iou, expected_bde, expected_acc = expected
    with (RESULTS / f"boxes_{cell}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    explanations = json.loads((RESULTS / f"explanations_{cell}.json").read_text(encoding="utf-8"))
    assert len(rows) == count, f"{cell}: CSV count {len(rows)} != {count}"
    assert len(explanations) == count, f"{cell}: JSON count {len(explanations)} != {count}"
    csv_by_id = {str(row["id"]): row for row in rows}
    assert len(csv_by_id) == count, f"{cell}: duplicate CSV ids"
    assert set(csv_by_id) == set(explanations), f"{cell}: JSON keys do not match CSV ids"
    for image_id, record in explanations.items():
        row = csv_by_id[image_id]
        csv_box = [float(row[name]) for name in ("x1", "y1", "x2", "y2")]
        assert record["box"] == csv_box, f"{cell}/{image_id}: JSON box != CSV box"
        assert not CJK.search(record["explanation"]), f"{cell}/{image_id}: CJK in explanation"
    mean_iou = sum(float(row["iou"]) for row in rows) / count
    mean_bde = sum(float(row["bde"]) for row in rows) / count
    acc_count = sum(float(row["iou"]) >= 0.75 for row in rows)
    assert round(mean_iou, 6) == expected_iou, f"{cell}: mIoU {mean_iou:.6f}"
    assert round(mean_bde, 6) == expected_bde, f"{cell}: BDE {mean_bde:.6f}"
    assert acc_count == expected_acc, f"{cell}: Acc@0.75 {acc_count}/{count}"
    print(f"PASS {cell}: {count} rows, mIoU={mean_iou:.6f}, BDE={mean_bde:.6f}, "
          f"Acc@0.75={acc_count}/{count}")


def compare_reproduced(directory):
    for cell in EXPECT:
        def read(base):
            with (base / f"boxes_{cell}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            by_id = {row["id"]: row for row in rows}
            assert len(by_id) == len(rows), f"{cell}: duplicate ids"
            return by_id
        reference, actual = read(RESULTS), read(Path(directory))
        assert reference.keys() == actual.keys(), f"{cell}: id mismatch"
        for image_id, row in reference.items():
            assert row.keys() == actual[image_id].keys(), f"{cell}: column mismatch"
            for column in row.keys() - {"id"}:
                assert math.isclose(float(row[column]), float(actual[image_id][column]),
                                    rel_tol=0, abs_tol=1e-9), (cell, image_id, column)
        print(f"PASS {cell}: reproduced values within 1e-9")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduced-dir", type=Path)
    args = parser.parse_args()
    if args.reproduced_dir:
        compare_reproduced(args.reproduced_dir)
    for cell, expected in EXPECT.items():
        check_cell(cell, expected)
    print("PASS: 1,690 explanation records across 4 JSON files; no CJK detected.")


if __name__ == "__main__":
    main()
