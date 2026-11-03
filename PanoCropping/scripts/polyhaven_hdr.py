#!/usr/bin/env python3
"""Fetch the Poly Haven HDR behind one of the 100 user-study panoramas (standard library only).

    python3 PanoCropping/scripts/polyhaven_hdr.py <scene | panorama.jpg> [--dir DIR]

The study rendered each working view and final shot from the 8k Poly Haven HDR with per-view
exposure (run_pano.py --hdr). PanoCropping/polyhaven_hdr.json pins, per scene, the HDR's URL, MD5
and size (all CC0) and the SHA-256 of the 8-bit panorama in PolyHaven_panoramas_100.zip, so a
panorama is matched by content, not by file name. Files are cached in DIR (default
$PHOTOGUIDER_HDR_DIR or ~/.cache/photoguider/hdr) and checked against the pinned MD5. The manifest
also records the coarse-scan direction each study shot was taken from (study_direction).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "polyhaven_hdr.json"


def _scenes():
    return json.loads(MANIFEST.read_text())["scenes"]


def default_dir():
    return Path(os.environ.get("PHOTOGUIDER_HDR_DIR", Path.home() / ".cache" / "photoguider" / "hdr"))


def scene_of(panorama):
    """The study scene whose 8-bit panorama has this file's SHA-256, or None."""
    digest = hashlib.sha256(Path(panorama).read_bytes()).hexdigest()
    return next((s for s, e in _scenes().items() if e["jpg_sha256"] == digest), None)


def study_direction(scene):
    """(pan, tilt) in degrees of the direction the user study shot this scene from."""
    d = _scenes()[scene]["direction"]
    return d["pan"], d["tilt"]


def fetch(scene, dest=None, log=print):
    """Path of the scene's verified HDR, downloading it first if needed."""
    entry = _scenes()[scene]
    dest = Path(dest or default_dir()); dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{scene}_8k.hdr"
    if path.is_file() and path.stat().st_size == entry["size"]:
        return path
    log(f"[hdr] downloading {scene} ({entry['size'] / 1e6:.0f} MB) from Poly Haven")
    tmp = path.with_suffix(".part")
    md5 = hashlib.md5()
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "PhotoGuider"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk); md5.update(chunk)
    if md5.hexdigest() != entry["md5"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{scene}: downloaded HDR does not match the pinned MD5")
    tmp.replace(path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Fetch the Poly Haven HDR of a user-study panorama.")
    ap.add_argument("scene", help="scene name (e.g. royal_esplanade) or a panorama file from PolyHaven_panoramas_100.zip")
    ap.add_argument("--dir", help="cache directory (default: $PHOTOGUIDER_HDR_DIR or ~/.cache/photoguider/hdr)")
    a = ap.parse_args()
    scene = a.scene if a.scene in _scenes() else scene_of(a.scene) if Path(a.scene).is_file() else None
    if scene is None:
        raise SystemExit(f"{a.scene} is not one of the 100 user-study panoramas")
    print(fetch(scene, a.dir))


if __name__ == "__main__":
    main()
