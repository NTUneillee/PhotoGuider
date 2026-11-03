"""Verify external files before use; invalid pins and mismatched files are errors."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent

def pins():
    return json.loads((HERE / "asset_pins.json").read_text())

def revision(name):
    value = pins()["revisions"][name]
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"Invalid pinned {name} commit in envs/asset_pins.json")
    return value

def verify(venus=False, require_present=False):
    inventory = pins()["venus_files" if venus else "files"]
    base = RELEASE / "envs/venus/weights/Venus-Q-Stage2" if venus else RELEASE
    for name, expected in inventory.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid expected SHA-256 for {name} in envs/asset_pins.json")
        path = base / name
        if not path.is_file():
            if require_present:
                raise FileNotFoundError(f"required model asset missing: {path}")
            print(f"WARN: not downloaded: {path}")
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch: {path}: {digest}")
        print(f"{name}: OK")
    if venus and base.exists():
        actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and ".cache" not in p.parts}
        if actual != set(inventory):
            raise ValueError(f"Venus inventory mismatch: missing={set(inventory)-actual}, extra={actual-set(inventory)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision")
    ap.add_argument("--venus", action="store_true")
    ap.add_argument("--require-present", action="store_true")
    args = ap.parse_args()
    if args.revision:
        print(revision(args.revision))
    else:
        verify(args.venus, args.require_present)
