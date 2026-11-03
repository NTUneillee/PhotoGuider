#!/usr/bin/env python3
"""CPU tests at the seams introduced by the inference-only revision:
pano.select_hotspots / crop_to_camera / final_size, explain.mask_geometry / build_prompt,
and the public result.json keys against OUTPUT_SCHEMA.md."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "inference" / "_lib")]
import pano  # noqa: E402
import explain as EX  # noqa: E402


def synthetic_scores(peaks):
    """Gaussian bumps at (pan, tilt, height) on the coarse grid."""
    rows = []
    for t in pano.SCAN_TILTS:
        for p in pano.SCAN_PANS:
            s = 0.0
            for pp, tt, h in peaks:
                s += h * np.exp(-(pano.gc_angle(p, t, pp, tt) ** 2) / (2 * 20.0 ** 2))
            rows.append(dict(pan=p, tilt=t, comp_score=s))
    return pd.DataFrame(rows)


def test_hotspots():
    hs = pano.select_hotspots(synthetic_scores([(90, 0, 1.0)]))
    assert hs and abs(hs[0][0] - 90) <= 2 and abs(hs[0][1]) <= 2, hs
    hs = pano.select_hotspots(synthetic_scores([(90, 0, 1.0), (270, 15, 0.95), (180, -15, 0.3)]))
    assert 1 <= len(hs) <= pano.K_MAX and abs(hs[0][0] - 90) <= 2, hs            # strongest first
    assert any(abs(p - 270) <= 2 for p, _, _ in hs), hs                           # second band peak kept
    assert all(abs(p - 180) > 5 for p, _, _ in hs), hs                            # weak one below the band
    hs = pano.select_hotspots(synthetic_scores([(0, 0, 1.0)]))                     # pan wrap at the seam
    assert hs and min(hs[0][0], 360 - hs[0][0]) <= 2, hs
    try:
        pano.select_hotspots(synthetic_scores([(0, 0, 1.0)]).iloc[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("missing grid cell must raise")
    print("select_hotspots on synthetic fields: PASS")


def test_camera():
    full = pano.crop_to_camera([0, 0, pano.W0, pano.H0], 37.0, -12.0)
    assert abs(full["fov"] - pano.FOV0) < 0.2 and abs(full["pan"] - 37) < 0.1 and abs(full["tilt"] + 12) < 0.1, full
    cx, cy = (pano.W0 - 1) / 2.0, (pano.H0 - 1) / 2.0          # the renderer's principal point
    half = [cx - pano.W0 / 4, cy - pano.H0 / 4, cx + pano.W0 / 4, cy + pano.H0 / 4]
    cam = pano.crop_to_camera(half, 200.0, 0.0)                 # pan 200 -> signed -160 (schema range)
    exp = 2 * np.degrees(np.arctan(np.tan(np.deg2rad(pano.FOV0 / 2)) / 2))
    assert abs(cam["fov"] - exp) < 1e-6 and abs(cam["pan"] + 160) < 1e-6 and abs(cam["tilt"]) < 1e-6, cam
    assert abs(cam["aspect"] - pano.W0 / pano.H0) < 1e-9
    assert pano.final_size(1.5) == (1920, 1280) and pano.final_size(0.5) == (960, 1920)
    print("crop_to_camera / final_size: PASS")


def test_geometry_prompt():
    W, H = 400, 300
    mask = np.zeros((30, 40), np.float32); mask[10:20, 10:20] = 1.0          # subject block
    g = EX.mask_geometry(mask, [0, 0, W, H], W, H)
    assert g["mask_ok"] and abs(g["capture"] - 1.0) < 1e-6 and abs(g["lost"]) < 1e-6, g
    g2 = EX.mask_geometry(mask, [0, 0, W * 15 / 40, H], W, H)                 # cuts the block in half
    assert abs(g2["capture"] - 0.5) < 1e-6, g2
    assert EX.mask_geometry(None, [0, 0, W, H], W, H) == {"mask_ok": False}
    assert EX.mask_geometry(np.zeros((30, 40), np.float32), [0, 0, W, H], W, H) == {"mask_ok": False}
    scores = dict(S=81, C=64, A=70, I=55)
    p, slot = EX.build_prompt(scores, "Kept the subject.", dict(g, sym_h_diff=0.02, cx=0.5, cy=0.5))
    assert "Subject=81" in p and "Integrity=55" in p and "System overview" in p and "cut off" in p
    assert "no geometric measurements" not in p.lower()
    p2, _ = EX.build_prompt(scores, "", {"mask_ok": False})
    assert "System overview" not in p2 and "no geometric measurements" in p2.lower()
    assert p.endswith(EX.SINGLE_STATEMENT) and p2.endswith(EX.SINGLE_STATEMENT)
    print("mask_geometry / build_prompt: PASS")


def test_schema_keys():
    doc = (ROOT / "OUTPUT_SCHEMA.md").read_text()
    top = re.findall(r"^\| `([a-z_]+)` \|", doc.split("`geometry.mask_ok`")[0], re.M)
    assert set(top) == {"input", "mode", "aspect", "box", "scores", "geometry", "explanation", "panorama", "versions"}, top
    geo = re.findall(r"^\| `([a-z_]+)`(?:, `([a-z_]+)`)? \|", doc.split("`geometry.mask_ok`")[1].split("For panoramas")[0], re.M)
    geo = {k for pair in geo for k in pair if k}
    W, H = 400, 300
    mask = np.zeros((30, 40), np.float32); mask[10:20, 10:20] = 1.0
    live = set(EX.mask_geometry(mask, [0, 0, W, H], W, H)) | {"sym_h_diff"}
    assert live == geo | {"mask_ok"}, (live, geo)
    print("OUTPUT_SCHEMA keys match live geometry: PASS")


def test_asset_pins():
    import model_pins                                   # extractors pin every HF backbone through it
    for name in ("clip", "qwen"):
        assert re.fullmatch(r"[0-9a-f]{40}", model_pins.revision(name)), name
    print("asset_pins.json resolves from inference/_lib: PASS")


def test_hotspots_reject_flat_and_unscored_fields():
    grid = [dict(pan=p, tilt=t) for t in pano.SCAN_TILTS for p in pano.SCAN_PANS]
    for value, message in ((0.0, "view scores are flat"), (float("nan"), "unscored")):
        rows = pd.DataFrame([dict(r, comp_score=value) for r in grid])
        try:
            pano.select_hotspots(rows)
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"flat/NaN field accepted: {value}")
    print("flat and NaN score fields rejected: PASS")


if __name__ == "__main__":
    test_hotspots(); test_camera(); test_geometry_prompt(); test_schema_keys(); test_asset_pins()
    test_hotspots_reject_flat_and_unscored_fields()
