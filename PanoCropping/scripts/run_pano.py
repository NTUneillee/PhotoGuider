#!/usr/bin/env python3
"""Choose and render a framing from one equirectangular panorama.

    envs/main/venv/bin/python PanoCropping/scripts/run_pano.py panorama.jpg --out output/pano [--mode full] [--explain]
        [--hdr panorama.hdr | --hdr auto [--scan-direction]]

Stage (a) direction: 120 perspective views (24 pans x 5 tilts, fov 40) are rendered and scored by
the additive view scorer; the strongest hotspot sets the shooting direction. Stage (b)-(d): the
60-degree working view in that direction goes through the photo pipeline (crop_image). The chosen
box is converted to camera parameters and the photograph is re-rendered natively from the panorama.
With --hdr (the linear source of the same panorama, as in the user study) the working view and the
final shot are rendered from the HDR and exposed for that view; the coarse scan always uses the
8-bit panorama. With --hdr auto a user-study panorama is also shot from the direction the study
used (polyhaven_hdr.json): its coarse scan ran in an environment that no longer exists, and on
near-tied hotspots it picked a different one than a scan today may. The scan still runs and its
hotspots are reported; --scan-direction shoots from this run's strongest hotspot instead.
Writes working_view.png, final.png and result.json (OUTPUT_SCHEMA.md) into --out.
"""
import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
INF = RELEASE / "inference"
sys.path[:0] = [str(HERE), str(INF), str(INF / "_lib")]
from pano import (SCAN_PANS, SCAN_TILTS, SCAN_FOV, SCAN_SIZE, FOV0, W0, H0,  # noqa: E402
                  render_view, render_hdr_view, load_hdr, select_hotspots, crop_to_camera, final_size)
import score_views as score  # noqa: E402
from photoguider import crop_image, Session  # noqa: E402


def scan(pano, coarse_dir, runner):
    """Score every view of the coarse grid. Returns a DataFrame (pan, tilt, comp_score)."""
    rows = []
    for tilt in SCAN_TILTS:
        for pan in SCAN_PANS:
            path = coarse_dir / f"p{pan:03d}_t{tilt:+03d}.jpg"
            rows.append({"vid": len(rows), "view": path.name, "path": str(path), "pan": pan, "tilt": tilt})

    def render(r):
        cv2.imwrite(r["path"], render_view(pano, r["pan"], r["tilt"], SCAN_FOV, SCAN_SIZE, SCAN_SIZE))
    with ThreadPoolExecutor(max_workers=4) as ex:      # each view is independent; same pixels as the loop
        list(ex.map(render, rows))
    maps = score.phase_maps(rows)
    with tempfile.TemporaryDirectory(prefix="features-", dir=coarse_dir) as scratch:
        frame = score.phase_deep(rows, maps, Path(scratch), runner=runner)
    raw, _mos = score.score_60f(score.build_scorer(), frame)
    assert len(frame) == len(raw), "deep-feature rows and scores differ"
    scored = frame[["photo_id"]].copy(); scored["comp_score"] = raw
    assert scored["photo_id"].is_unique, "duplicate scored photo_id"
    meta = pd.DataFrame([{"photo_id": r["vid"], "pan": r["pan"], "tilt": r["tilt"]} for r in rows])
    merged = meta.merge(scored, on="photo_id", how="inner", validate="one_to_one")
    assert len(merged) == len(rows), "every rendered view must be scored exactly once"
    return merged.drop(columns="photo_id")


def run(pano_path, mode, explain, out, hdr_path=None, scan_direction=False, session=None):
    if explain:
        from env_config import require_env
        require_env("explainer")                    # fail before the coarse scan
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    pano_path = Path(pano_path).resolve()
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise SystemExit(f"cannot read equirectangular panorama: {pano_path}")
    h, w = pano.shape[:2]
    if abs(w / h - 2) > 0.02:
        raise SystemExit(f"expected a 2:1 equirectangular panorama, got {w}x{h}: {pano_path}")
    hdr = study = None
    if hdr_path == "auto":                          # a user-study panorama: fetch its pinned Poly Haven HDR
        from polyhaven_hdr import scene_of, fetch, study_direction
        scene = scene_of(pano_path)
        hdr_path = fetch(scene) if scene else None
        study = study_direction(scene) if scene and not scan_direction else None
        print(f"[hdr] {scene}: per-view exposure from {hdr_path}" if scene else
              "[hdr] not a user-study panorama; rendering the working view from the panorama itself")
    if hdr_path is not None:
        hdr = load_hdr(Path(hdr_path).resolve())
        if hdr.shape[1] * h != hdr.shape[0] * w:
            raise SystemExit(f"the HDR ({hdr.shape[1]}x{hdr.shape[0]}) must be the same panorama as {pano_path} ({w}x{h})")
    coarse_dir = out / "_coarse"; coarse_dir.mkdir(exist_ok=True)
    fine_dir = out / "_fine"
    own = session is None
    session = session or Session(verbose=True)
    try:
        scores = scan(pano, coarse_dir, session.pool.run_op)
        hotspots = select_hotspots(scores, SCAN_PANS, SCAN_TILTS)
        if not hotspots:
            raise RuntimeError("coarse scan returned no shooting direction")
        if study is not None:                       # shoot from the study's direction, listed first
            row = scores[(scores["pan"] == study[0]) & (scores["tilt"] == study[1])]
            assert len(row) == 1, f"study direction {study} is not on the scan grid"
            rank = next((i + 1 for i, hs in enumerate(hotspots) if (hs[0], hs[1]) == study), None)
            hotspots = [(*study, float(row["comp_score"].iloc[0]))] + [hs for hs in hotspots if (hs[0], hs[1]) != study]
            print(f"[direction] user-study direction pan {study[0]:g}, tilt {study[1]:g}; this scan ranks it "
                  + (f"#{rank}" if rank else "outside its hotspots"))
        pan0, tilt0, _ = hotspots[0]
        if hdr is not None:
            working, gain = render_hdr_view(hdr, pan0, tilt0, FOV0, W0, H0)
        else:
            working, gain = render_view(pano, pan0, tilt0, FOV0, W0, H0), None
        cv2.imwrite(str(out / "working_view.png"), working)
        fine = crop_image(out / "working_view.png", None, mode, explain, fine_dir, session=session)
    finally:
        if own:
            session.close()
    cam = crop_to_camera(fine["box"], pan0, tilt0)
    ow, oh = final_size(cam["aspect"])
    if hdr is not None:
        final, _ = render_hdr_view(hdr, cam["pan"], cam["tilt"], cam["fov"], ow, oh, gain=gain)
    else:
        final = render_view(pano, cam["pan"], cam["tilt"], cam["fov"], ow, oh)
    cv2.imwrite(str(out / "final.png"), final)
    result = dict(fine)
    result["input"] = str(pano_path)
    result["panorama"] = {
        "pan": round(cam["pan"], 3), "tilt": round(cam["tilt"], 3), "fov": round(cam["fov"], 3),
        "width": ow, "height": oh, "aspect": ow / oh,
        "exposure": {"source": "hdr", "gain": round(float(gain), 4)} if hdr is not None else {"source": "panorama"},
        "direction": "study" if study is not None else "scan",
        "hotspots": [{"pan": round(p, 2), "tilt": round(t, 2), "score": round(float(s), 4)} for p, t, s in hotspots],
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    shutil.rmtree(coarse_dir, ignore_errors=True)
    shutil.rmtree(fine_dir, ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Choose and render a framing from one equirectangular panorama.")
    ap.add_argument("pano", help="input equirectangular panorama image")
    ap.add_argument("--mode", choices=("fast", "full"), default="fast",
                    help="fine-crop mode (default: fast; full enables the consultant)")
    ap.add_argument("--explain", action="store_true", help="add a written explanation (GPU explainer environment required)")
    ap.add_argument("--hdr", help="the linear .hdr of the same panorama: the working view and final shot are then "
                                  "rendered from it with per-view exposure (the user-study setting); 'auto' "
                                  "fetches it for the 100 user-study panoramas (polyhaven_hdr.py)")
    ap.add_argument("--scan-direction", action="store_true",
                    help="with --hdr auto, shoot a user-study panorama from this run's strongest hotspot instead "
                         "of the direction the study used")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    run(a.pano, a.mode, a.explain, a.out, hdr_path=a.hdr, scan_direction=a.scan_direction)


if __name__ == "__main__":
    main()
