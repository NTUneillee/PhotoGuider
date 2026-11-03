"""PhotoGuider: one photograph in, one framed crop out, with its four assessments and (optionally)
a written explanation.

    from photoguider import crop_image
    result = crop_image("photo.jpg", aspect="16:9", mode="full", explain=True, out="output/photo")

Writes ``crop.jpg`` and ``result.json`` (OUTPUT_SCHEMA.md) into ``out`` and returns the result dict.
``mode="fast"`` runs the ranker + gated refinement only; ``"full"`` adds the Venus consultant
(its environment and weights must be installed). ``explain=True`` needs the explainer environment
and adapter. Keep the same ``Session`` across many images so the extractor workers stay resident.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
LIB = HERE / "_lib"
sys.path[:0] = [str(HERE), str(LIB)]
import infer as pipeline                                     # noqa: E402  the crop pipeline
import explain as EX                                         # noqa: E402  assessments + prompt + VLM
from feature_extractors import Extractors                    # noqa: E402
from workers import ExtractorPool                            # noqa: E402
from photog_feats import sym_h_diff                          # noqa: E402
from image_io import open_upright                            # noqa: E402
from aspect import parse_aspect, fit_aspect                  # noqa: E402
from env_config import require_env                           # noqa: E402

MAX_SIDE = 2048     # very large inputs run on a bounded working copy; boxes are mapped back


def versions():
    try:
        code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        code = "unknown"
    adapter = EX.ADAPTER_DIR / "adapter_model.safetensors"
    sha = None
    if adapter.is_file():
        with adapter.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"code": code, "adapter_sha": sha}


class Session:
    """Resident extractor workers + the loaded ranker and calibration tables, shared across images."""

    def __init__(self, verbose=False):
        self.pool = ExtractorPool(verbose=verbose)
        self.ranker = EX.load_ranker()
        self.feat2tok = EX.load_groups()
        self.table = EX.load_calibration()
        self.versions = versions()          # hashed once per session; the adapter file is ~150 MB

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def crop_image(image, aspect, mode, explain, out, session=None, strict_aspect=True):
    if mode not in ("fast", "full"):
        raise ValueError(f"mode must be 'fast' or 'full', got {mode!r}")
    if aspect is not None:
        try:
            parse_aspect(aspect)                      # fail before any model loads
        except ValueError as exc:
            raise SystemExit(f"invalid --aspect {aspect!r}: {exc} (examples: 16:9, 4:5, 1)") from exc
    if explain:
        require_env("explainer")                      # fail before the expensive crop stage
    image = Path(image).resolve()
    if not image.is_file():
        raise SystemExit(f"input image not found: {image}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    own = session is None
    s = session or Session()
    try:
        with Extractors.temporary(runner=s.pool.run_op, release=s.pool.close) as ex:
            return _crop_image(s, ex, image, aspect, mode, explain, out, strict_aspect)
    finally:
        if own:
            s.close()


def _crop_image(s, ex, image, aspect, mode, explain, out, strict_aspect):
    orig, rotated = open_upright(image)
    scale, work = 1.0, image
    if rotated or max(orig.size) > MAX_SIDE:
        # every stage reads the image from disk, so it gets one upright, bounded, lossless copy
        scale = min(1.0, MAX_SIDE / max(orig.size))
        work = out / "_work.png"
        w = orig.convert("RGB")
        if scale < 1.0:
            w = w.resize((round(orig.size[0] * scale), round(orig.size[1] * scale)), Image.LANCZOS)
        w.save(work)
        print(f"[input] {orig.size[0]}x{orig.size[1]}" + (" EXIF-rotated" if rotated else "")
              + f" -> working copy {w.size[0]}x{w.size[1]}")
    try:
        # 1. the crop (pool -> shortlist -> ranker -> refinement -> consultant)
        res, ctx = pipeline.run(str(work), aspect, mode == "full", str(out / "_consultant"), ex)
        final = res["full"]["final"] if mode == "full" else res["fast"]["final"]
        top1 = res["full"]["winner"] if mode == "full" else res["fast"]["top1"]
        W, H = ctx["W"], ctx["H"]
        if aspect is not None and strict_aspect:
            # the benchmark protocol clips a border-expanded box per side; deliver the exact ratio instead
            final = fit_aspect(final, parse_aspect(aspect), W, H)
        # 2. the four assessments on the final box
        frame = EX.featurize_final(pipeline, ex, [r["box"] for r in res["k10"]], final, 0, work, W, H,
                                   ctx["vmix"], ctx["v1"], ctx["mask"])   # each box carries its own ratio
        scores, _, ov = EX.dashboard(s.ranker, s.feat2tok, s.table, frame, len(frame) - 1, top1, final)
        # 3. measured geometry + the explanation prompt (always built; the VLM only on request)
        mask01 = ctx["mask"].astype(np.float32) / 255.0 if ctx["mask"] is not None and ctx["mask"].max() > 1.5 else ctx["mask"]
        geometry = EX.mask_geometry(mask01, final, W, H)
        if geometry.get("mask_ok"):
            geometry["sym_h_diff"] = float(sym_h_diff(ctx["vmix"], *final, W, H))
        prompt, _ = EX.build_prompt(scores, ov, geometry)
        # 4. outputs
        final_orig = [x / scale for x in final]
        orig.convert("RGB").crop((int(final_orig[0]), int(final_orig[1]), int(final_orig[2]), int(final_orig[3]))).save(out / "crop.jpg", quality=95)
        if explain:
            ex.release_gpu()                          # the explainer (~9 GB) loads on the same GPU
        paragraph = EX.write_explanation(out / "crop.jpg", prompt) if explain else None
        result = {
            "input": str(image), "mode": mode, "aspect": aspect,
            "box": [round(float(x), 1) for x in final_orig], "scores": scores,
            "geometry": geometry,
            "explanation": paragraph, "panorama": None, "versions": s.versions,
        }
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[result] S{scores['S']} C{scores['C']} A{scores['A']} I{scores['I']}  box {result['box']}")
        return result
    finally:
        if work != image:
            Path(work).unlink(missing_ok=True)
        shutil.rmtree(out / "_consultant", ignore_errors=True)
