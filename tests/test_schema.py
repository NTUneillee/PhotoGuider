#!/usr/bin/env python3
"""Validate the shared subset represented by every shipped explanation record."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(rec):
    assert set(("box", "scores", "geometry", "explanation")) <= set(rec)
    assert len(rec["box"]) == 4 and all(isinstance(x, (int, float)) for x in rec["box"])
    assert rec["box"][2] > rec["box"][0] and rec["box"][3] > rec["box"][1]
    assert set(rec["scores"]) == set("SCAI")
    assert all(isinstance(v, int) and 0 <= v <= 100 for v in rec["scores"].values())
    assert isinstance(rec["geometry"].get("mask_ok"), bool)
    assert rec["explanation"] is None or isinstance(rec["explanation"], str)


def main():
    total = 0
    for path in sorted((ROOT / "inference" / "results").glob("explanations_*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(records, dict) and records
        for rec in records.values():
            validate(rec)
            total += 1
    assert total == 1690, total
    print(f"schema shared subset: PASS ({total} records)")


if __name__ == "__main__":
    main()
