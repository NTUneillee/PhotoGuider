#!/usr/bin/env python3
"""PhotoGuider in the browser: crop a photograph or frame a view inside a 360° panorama.

    envs/gui/venv/bin/python app.py [--port 7860] [--share]

The GUI has its own small environment (envs/gui, Gradio only). Each request runs the command-line
pipeline (inference/run_image.py or PanoCropping/scripts/run_pano.py) in the main environment, so
results are identical to the CLI, and streams its progress into the page. Requests run one at a
time on the GPU.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
MAIN_PY = ROOT / "envs" / "main" / "venv" / "bin" / "python"
RUN_IMAGE = ROOT / "inference" / "run_image.py"
RUN_PANO = ROOT / "PanoCropping" / "scripts" / "run_pano.py"
# the eight panoramas ranked highest in the 360° user study (also in PolyHaven_panoramas_100.zip)
PANO_EXAMPLES = [ROOT / "examples" / "panoramas" / f"{s}.jpg"
                 for s in ("royal_esplanade", "golden_bay", "neurathen_rock_castle", "dam_wall", "preller_drive",
                           "konigsallee", "shanghai_bund", "decor_shop")]
# the six landscape FCDB / FLMS test photographs on which PhotoGuider's crop is closest to the human crop
PHOTO_EXAMPLES = [ROOT / "examples" / "photos" / f"{s}.jpg"
                  for s in ("FLMS_535530800", "FCDB_1149089847", "FLMS_6854965228", "FLMS_6856279866",
                            "FCDB_185452253", "FLMS_1108687475")]

RATIOS = ["1:1", "4:5", "3:2", "4:3", "16:9", "9:16"]
TOKENS = [("S", "Subject"), ("C", "Composition"), ("I", "Integrity"), ("A", "Aesthetics")]
DISPLAY_LONG = 1600
# free GPU memory (GB) each configuration needs; below it the run is likely to fail mid-way
NEED_GB = {("fast", False): 5, ("fast", True): 10, ("full", False): 11, ("full", True): 11}


# ───────────────────────────── running the pipeline ─────────────────────────────
def free_gpu_gb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      text=True, timeout=10)
        return int(out.split()[0]) / 1024
    except Exception:
        return None


def check_gpu(mode, explain):
    free, need = free_gpu_gb(), NEED_GB[(mode, explain)]
    if free is not None and free < need:
        gr.Warning(f"The GPU has {free:.1f} GB free; this setting needs about {need} GB. "
                   "Another process is using the GPU, so the run may fail.")


def run_streaming(cmd):
    """Run one pipeline command; yield the growing log, then raise on a nonzero exit."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        yield "\n".join(lines[-400:])
    if proc.wait() != 0:
        tail = "\n".join(lines[-12:])
        raise gr.Error(f"PhotoGuider stopped with an error:\n{tail}", duration=None)


def new_out_dir():
    return Path(tempfile.mkdtemp(prefix="photoguider_"))


# ───────────────────────────── result views ─────────────────────────────
def crop_overlay(image_path, box):
    """The input with everything outside the crop dimmed, the crop outlined and its thirds drawn."""
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")   # boxes are in upright pixels
    s = min(1.0, DISPLAY_LONG / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    x1, y1, x2, y2 = [v * s for v in box]
    mask = Image.new("L", img.size, 150)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=0)
    out = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, mask)
    d = ImageDraw.Draw(out)
    for k in (1, 2):
        xs, ys = x1 + (x2 - x1) * k / 3, y1 + (y2 - y1) * k / 3
        d.line([xs, y1, xs, y2], fill=(255, 255, 255), width=1)
        d.line([x1, ys, x2, ys], fill=(255, 255, 255), width=1)
    d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=max(2, round(max(img.size) / 400)))
    return out


def _view_outline(pano_w, pano_h, pan, tilt, fov, out_w, out_h, n=60):
    """Equirect pixel coordinates of a perspective view's border (same camera model as inference/_lib/pano.py)."""
    f = (out_w / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    u = np.linspace(0, out_w - 1, n) - (out_w - 1) / 2.0
    v = np.linspace(0, out_h - 1, n) - (out_h - 1) / 2.0
    top, bottom = np.stack([u, np.full(n, v[0])], 1), np.stack([u[::-1], np.full(n, v[-1])], 1)
    right, left = np.stack([np.full(n, u[-1]), v], 1), np.stack([np.full(n, u[0]), v[::-1]], 1)
    uv = np.concatenate([top, right, bottom, left])
    d = np.stack([uv[:, 0], -uv[:, 1], np.full(len(uv), f)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a, b = np.deg2rad(-tilt), np.deg2rad(pan)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    w = d @ (ry @ rx).T
    lon, lat = np.arctan2(w[:, 0], w[:, 2]), np.arcsin(np.clip(w[:, 1], -1, 1))
    return (lon / (2 * np.pi) + 0.5) * pano_w, (0.5 - lat / np.pi) * pano_h


def pano_overlay(pano_path, cam):
    """The panorama with the chosen view's footprint and the scan's hotspots marked."""
    img = Image.open(pano_path).convert("RGB")
    s = min(1.0, 2048 / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    W, H = img.size
    d = ImageDraw.Draw(img)
    lw = max(3, W // 500)
    xs, ys = _view_outline(W, H, cam["pan"], cam["tilt"], cam["fov"], cam["width"], cam["height"])
    pts = list(zip(xs, ys))
    for (xa, ya), (xb, yb) in zip(pts, pts[1:] + pts[:1]):
        if abs(xb - xa) < W / 2:                      # skip the segment that wraps across the seam
            d.line([xa, ya, xb, yb], fill=(255, 255, 255), width=lw)
    for i, h in enumerate(cam.get("hotspots") or []):
        p = ((h["pan"] + 180) % 360) - 180
        x, y, r = (p / 360 + 0.5) * W, (0.5 - h["tilt"] / 180) * H, lw * 3
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=lw,
                  fill=(239, 182, 92) if i == 0 else None)
    return img


def scores_html(scores, cam=None, two=False):
    if not scores:
        return EMPTY_SCORES
    rows = "".join(
        f'<div class="pg-row pg-{k}"><span class="pg-dot">{k}</span><span class="pg-name">{name}</span>'
        f'<span class="pg-track"><span class="pg-fill" style="width:{int(scores[k])}%"></span></span>'
        f'<span class="pg-val">{int(scores[k])}</span></div>' for k, name in TOKENS)
    camera = ""
    if cam:
        def deg(x, signed=True):
            v = round(x)
            return f"{v:+d}°" if signed and v else f"{v}°"
        camera = ('<div class="pg-cam">'
                  f'<div><span>Pan</span><b>{deg(cam["pan"])}</b></div>'
                  f'<div><span>Tilt</span><b>{deg(cam["tilt"])}</b></div>'
                  f'<div><span>Field of view</span><b>{deg(cam["fov"], signed=False)}</b></div></div>')
    if two:
        return f'<div class="pg-card pg-two">{rows}</div>'
    return f'<div class="pg-card">{camera}{rows}</div>'


def explanation_html(text):
    if not text:
        return EMPTY_TEXT
    body = html.escape(text).replace("\n\n", "</p><p>")
    return f'<div class="pg-card pg-text"><p>{body}</p></div>'


# ───────────────────────────── handlers ─────────────────────────────
def crop_photo(image_path, fixed, ratio, mode, explain):
    if not image_path:
        raise gr.Error("Choose a photograph first.")
    check_gpu(mode, explain)
    out = new_out_dir()
    cmd = [str(MAIN_PY), str(RUN_IMAGE), image_path, "--out", str(out), "--mode", mode]
    if fixed:
        cmd += ["--aspect", ratio]
    if explain:
        cmd.append("--explain")
    log = ""
    for log in run_streaming(cmd):
        yield None, None, EMPTY_SCORES, EMPTY_TEXT, gr.update(value=None, visible=False), log
    res = json.loads((out / "result.json").read_text())
    yield (crop_overlay(image_path, res["box"]), str(out / "crop.jpg"), scores_html(res["scores"], two=True),
           explanation_html(res["explanation"]), gr.update(value=str(out / "result.json"), visible=True), log)


def frame_pano(pano_path, mode, explain):
    if not pano_path:
        raise gr.Error("Choose an equirectangular panorama first.")
    check_gpu(mode, explain)
    out = new_out_dir()
    # the 100 user-study panoramas are rendered from their Poly Haven HDR, as in the study
    cmd = [str(MAIN_PY), str(RUN_PANO), pano_path, "--out", str(out), "--mode", mode, "--hdr", "auto"]
    if explain:
        cmd.append("--explain")
    log = ""
    for log in run_streaming(cmd):
        yield None, None, EMPTY_SCORES, EMPTY_TEXT, gr.update(value=None, visible=False), log
    res = json.loads((out / "result.json").read_text())
    yield (pano_overlay(pano_path, res["panorama"]), str(out / "final.png"), scores_html(res["scores"], res["panorama"]),
           explanation_html(res["explanation"]), gr.update(value=str(out / "result.json"), visible=True), log)


# ───────────────────────────── page ─────────────────────────────
# grouped-list layout in the iOS manner; accent and token colours follow the project page
CSS = """
:root {--pg-bg: #f2f2f7; --pg-card: #ffffff; --pg-ink: #1c1c1e; --pg-ink2: #6e6e73; --pg-sep: #e5e5ea;
       --pg-fill: rgba(118,118,128,.12); --pg-accent: #2e6da4; --pg-on: #34c759;
       --pg-S: #e8735a; --pg-C: #5b8fd6; --pg-I: #b78ac9; --pg-A: #63b28a}
.gradio-container {width: 100% !important; max-width: 1560px !important; margin: 0 auto !important; padding: 32px 40px 48px !important}
.pg-main {gap: 44px !important}
footer {display: none !important}
.pg-bare, .pg-bare > div, .pg-bare .prose {background: transparent !important; border: none !important; padding: 0 !important;
          box-shadow: none !important; min-height: 0 !important; overflow: visible !important}

/* large title */
.pg-title {padding: 0 4px 2px}
.pg-title h1 {margin: 0; font-size: 34px; font-weight: 700; letter-spacing: -0.02em; color: var(--pg-ink)}
.pg-title p {margin: 6px 0 0; font-size: 19px; color: var(--pg-ink2)}

/* segmented control: tabs and radios */
.pg-tabs .tab-wrapper {border: none !important; width: 100%; max-width: 360px; margin: 14px 0 16px 0 !important}
.pg-tabs .tab-container[role=tablist] {display: flex; gap: 0; padding: 2px; border-radius: 9px;
                                        background: var(--pg-fill); border: none !important}
.pg-tabs .tab-container[role=tablist] button {flex: 1; border: none !important; border-radius: 7px; padding: 6px 12px;
                                              font-size: 14px; font-weight: 500; color: var(--pg-ink) !important}
.pg-tabs .tab-container[role=tablist] button.selected {background: #fff; font-weight: 600;
                                              box-shadow: 0 3px 8px rgba(0,0,0,.12), 0 3px 1px rgba(0,0,0,.04)}
.pg-tabs .tab-container[role=tablist] button.selected::after {display: none !important}
.pg-seg .wrap {display: inline-flex !important; gap: 0 !important; padding: 2px; border-radius: 9px;
               background: var(--pg-fill); flex-wrap: nowrap !important}
.pg-seg label {border: none !important; background: transparent !important; box-shadow: none !important;
               border-radius: 7px !important; padding: 5px 14px !important; margin: 0 !important; min-width: 0 !important}
.pg-seg label.selected {background: #fff !important; box-shadow: 0 3px 8px rgba(0,0,0,.12), 0 3px 1px rgba(0,0,0,.04) !important}
.pg-seg label input {position: absolute; opacity: 0; pointer-events: none}
.pg-seg label span {margin: 0 !important; font-size: 14px; color: var(--pg-ink)}
.pg-seg label.selected span {font-weight: 600}
.pg-row-ctl {display: flex !important; align-items: center; justify-content: space-between; gap: 12px}
.pg-row-ctl > span[data-testid=block-info] {margin: 0 !important; font-size: 16px; color: var(--pg-ink)}

/* grouped list */
.pg-head {font-size: 13px; letter-spacing: .02em; text-transform: uppercase; color: var(--pg-ink2);
          padding: 0 16px; margin: 4px 0 -12px}
.pg-group {background: var(--pg-card) !important; border-radius: 14px !important; border: none !important;
           overflow: hidden; box-shadow: none !important; gap: 0 !important}
.pg-group > div, .pg-group .block {background: var(--pg-card) !important; border: none !important; border-radius: 0 !important;
                                   box-shadow: none !important}
.pg-group > div + div {border-top: 1px solid var(--pg-sep) !important; margin-left: 16px}
.pg-group .block {padding: 11px 16px 11px 0 !important}
.pg-group > div:first-child .block, .pg-group > div:first-child.block {padding-left: 16px !important}
.pg-flush, .pg-flush .block {padding: 0 !important}
.pg-group > div.pg-flush + div {margin-left: 16px}

/* toggle switch */
.pg-switch label {display: flex !important; flex-direction: row-reverse; justify-content: space-between;
                  align-items: center; width: 100%; gap: 12px; cursor: pointer}
.pg-switch label span {font-size: 16px; color: var(--pg-ink); margin: 0 !important}
.pg-switch input[type=checkbox] {appearance: none; -webkit-appearance: none; width: 51px !important; height: 31px !important;
     border-radius: 999px !important; background: #e9e9ea !important; background-image: none !important; border: none !important;
     position: relative; flex: none; margin: 0 !important; transition: background .2s; box-shadow: none !important}
.pg-switch input[type=checkbox]::before {content: ""; position: absolute; top: 2px; left: 2px; width: 27px; height: 27px;
     border-radius: 50%; background: #fff; box-shadow: 0 3px 8px rgba(0,0,0,.15), 0 3px 1px rgba(0,0,0,.06); transition: left .2s}
.pg-switch input[type=checkbox]:checked {background: var(--pg-on) !important}
.pg-switch input[type=checkbox]:checked::before {left: 22px}

/* primary button */
.pg-go {background: var(--pg-accent) !important; color: #fff !important; border: none !important; border-radius: 12px !important;
        font-size: 17px !important; font-weight: 600 !important; padding: 13px !important; margin-top: 6px; box-shadow: none !important}
.pg-go:hover {filter: brightness(1.08)}

/* result cards */
.pg-media {background: var(--pg-card) !important; border-radius: 14px !important; border: none !important; overflow: hidden}
.pg-media .block-label, .pg-group .block-label {background: rgba(255,255,255,.86) !important; border: none !important;
     border-radius: 8px !important; color: var(--pg-ink2) !important; box-shadow: none !important; font-size: 12px}
.pg-bare .pg-card, .pg-card {background: var(--pg-card) !important; border-radius: 14px; padding: 6px 16px !important}
.pg-row {display: grid; grid-template-columns: 26px 112px 1fr 34px; align-items: center; gap: 10px; padding: 11px 0}
.pg-row + .pg-row {border-top: 1px solid var(--pg-sep)}
.pg-two {display: grid; grid-template-columns: 1fr 1fr; column-gap: 28px}
.pg-two .pg-row {border-top: none} .pg-two .pg-row:nth-child(n+3) {border-top: 1px solid var(--pg-sep)}
/* a result card stretched to the height of the image beside it */
.pg-eq .pg-bare:not(.pg-fillh) {flex: none !important}
.pg-eq .pg-fillh {flex: 1 1 auto !important; display: flex !important; flex-direction: column; gap: 0 !important}
.pg-eq .pg-fillh > div {flex: 1 1 auto; display: flex; flex-direction: column}
.pg-eq .pg-fillh .pg-card, .pg-eq .pg-fillh .pg-empty {flex: 1 1 auto; display: flex; flex-direction: column;
                                                      justify-content: space-around}
.pg-dot {width: 24px; height: 24px; border-radius: 7px; display: grid; place-items: center; color: #fff;
         font-size: 12px; font-weight: 700}
.pg-name {font-size: 15px; color: var(--pg-ink)}
.pg-track {height: 6px; border-radius: 999px; background: var(--pg-fill); overflow: hidden}
.pg-fill {display: block; height: 100%; border-radius: 999px}
.pg-val {text-align: right; font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--pg-ink)}
.pg-S .pg-dot, .pg-S .pg-fill {background: var(--pg-S)} .pg-C .pg-dot, .pg-C .pg-fill {background: var(--pg-C)}
.pg-I .pg-dot, .pg-I .pg-fill {background: var(--pg-I)} .pg-A .pg-dot, .pg-A .pg-fill {background: var(--pg-A)}
.pg-cam {display: grid; grid-template-columns: repeat(3, 1fr); padding: 12px 0; border-bottom: 1px solid var(--pg-sep)}
.pg-cam div {display: flex; flex-direction: column; gap: 2px}
.pg-cam span {font-size: 12px; color: var(--pg-ink2)} .pg-cam b {font-size: 17px; font-weight: 600; color: var(--pg-ink)}
.pg-bare .pg-text {padding: 14px 16px !important} .pg-text p {margin: 0 0 .6em; font-size: 15px; line-height: 1.55; color: var(--pg-ink)}
.pg-text p:last-child {margin: 0}
.pg-foot {font-size: 13px; line-height: 1.45; color: var(--pg-ink2); padding: 0 16px; margin-top: -6px}
.pg-foot a {color: var(--pg-accent) !important; text-decoration: none}
.pg-examples, .pg-examples .block {background: transparent !important; border: none !important; padding: 0 !important}
.pg-examples .label {display: none !important}
.pg-examples .gallery {display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px !important}
.pg-ex-pano .gallery {grid-template-columns: repeat(4, minmax(0, 1fr))}
.pg-examples .gallery-item {width: 100% !important; padding: 0 !important; margin: 0 !important; border: none !important;
                            border-radius: 10px !important; overflow: hidden; background: none !important}
.pg-examples .gallery-item .container {display: block !important; width: 100% !important; max-width: none !important;
                                       height: auto !important; aspect-ratio: 4 / 3; border: none !important;
                                       border-radius: 10px; overflow: hidden}
.pg-ex-pano .gallery-item .container {aspect-ratio: 2}
.pg-examples .gallery-item img {display: block; width: 100% !important; height: 100% !important; max-width: none !important;
                                object-fit: cover !important}
.pg-examples .gallery-item:hover {opacity: .85}
.pg-bare .pg-empty, .pg-empty {background: var(--pg-card) !important; border-radius: 14px; padding: 18px 16px !important; color: var(--pg-ink2); font-size: 15px}
.pg-log, .pg-log .block {background: transparent !important; border: none !important}
.pg-log > button {padding: 4px 16px !important; color: var(--pg-ink2) !important; font-size: 14px}
.pg-file .file-preview-holder, .pg-file {font-size: 14px}
"""

TITLE = """<div class="pg-title">
<h1>PhotoGuider</h1>
<p>Composition-Aware Cropping and 360° View Framing with Explainable Photographic Assessments</p>
</div>"""

PANOS_URL = "https://github.com/NTUneillee/PhotoGuider/releases/download/release/PolyHaven_panoramas_100.zip"
PANO_FOOT = (f'<div class="pg-foot">Equirectangular, 2:1. No panorama at hand? Download the 100 '
             f'<a href="{PANOS_URL}">Poly Haven panoramas</a> from our user study (CC0, 900 MB). For these '
             f'panoramas the matching Poly Haven HDR (about 100 MB) is fetched on first use, as in the study.</div>')
EMPTY_SCORES = '<div class="pg-empty">The four assessments appear here.</div>'
EMPTY_TEXT = '<div class="pg-empty">The written explanation appears here.</div>'


def head(text):
    gr.HTML(f'<div class="pg-head">{text}</div>', elem_classes="pg-bare")


def options(photo):
    """The OPTIONS group; for photos it starts with the fixed-ratio switch and its ratio picker."""
    fixed = ratio = None
    head("Options")
    with gr.Group(elem_classes="pg-group"):
        if photo:
            fixed = gr.Checkbox(value=False, label="Fixed aspect ratio", elem_classes="pg-switch")
            ratio = gr.Radio(RATIOS, value="16:9", show_label=False, visible=False, elem_classes="pg-seg")
        mode = gr.Radio(["fast", "full"], value="fast", label="Mode", elem_classes=["pg-seg", "pg-row-ctl"])
        explain = gr.Checkbox(value=True, label="Written explanation", elem_classes="pg-switch")
    if photo:
        fixed.change(lambda on: gr.update(visible=on), fixed, ratio)
    return fixed, ratio, mode, explain


def build():
    theme = gr.themes.Base(primary_hue="blue", neutral_hue="gray", radius_size="lg",
                           font=[gr.themes.Font("-apple-system"), gr.themes.Font("BlinkMacSystemFont"), gr.themes.GoogleFont("Inter"),
                                 gr.themes.Font("sans-serif")]).set(
        body_background_fill="#f2f2f7",
        body_text_color="#1c1c1e",
        body_text_color_subdued="#6e6e73",
        block_background_fill="#ffffff",
        block_border_width="0px",
        block_shadow="none",
        block_radius="14px",
        border_color_primary="#e5e5ea",
        color_accent="#2e6da4",
        color_accent_soft="#eaf1f8",
    )
    with gr.Blocks(title="PhotoGuider", theme=theme, css=CSS) as demo:
        gr.HTML(TITLE, elem_classes="pg-bare")
        with gr.Tabs(elem_classes="pg-tabs"):
            with gr.Tab("Photo"):
                with gr.Row(equal_height=False, elem_classes="pg-main"):
                    with gr.Column(scale=4, min_width=300):
                        head("Photograph")
                        with gr.Group(elem_classes="pg-group"):
                            p_in = gr.Image(type="filepath", show_label=False, height=240, elem_classes="pg-flush")
                        examples = [[str(p)] for p in PHOTO_EXAMPLES if p.is_file()]
                        if examples:
                            head("Examples")
                            with gr.Column(elem_classes="pg-examples"):
                                gr.Examples(examples, inputs=p_in, label="", examples_per_page=6)
                        p_fixed, p_ratio, p_mode, p_explain = options(photo=True)
                        p_run = gr.Button("Crop", elem_classes="pg-go")
                        p_json = gr.File(label="result.json", height=56, visible=False, elem_classes="pg-file")
                        with gr.Accordion("Progress log", open=False, elem_classes="pg-log"):
                            p_log = gr.Textbox(lines=10, max_lines=10, show_label=False, autoscroll=True)
                    with gr.Column(scale=7):
                        head("Result")
                        with gr.Row():
                            p_overlay = gr.Image(label="On the original", interactive=False, height=270,
                                                 elem_classes="pg-media")
                            p_crop = gr.Image(label="Cropped", interactive=False, height=270, elem_classes="pg-media")
                        head("Assessments")
                        p_scores = gr.HTML(EMPTY_SCORES, elem_classes="pg-bare")
                        head("Explanation")
                        p_text = gr.HTML(EMPTY_TEXT, elem_classes="pg-bare")
                p_run.click(crop_photo, [p_in, p_fixed, p_ratio, p_mode, p_explain],
                            [p_overlay, p_crop, p_scores, p_text, p_json, p_log], concurrency_id="gpu")

            with gr.Tab("360° Panorama"):
                with gr.Row(equal_height=False, elem_classes="pg-main"):
                    with gr.Column(scale=4, min_width=300):
                        head("Panorama")
                        with gr.Group(elem_classes="pg-group"):
                            v_in = gr.Image(type="filepath", show_label=False, height=180, elem_classes="pg-flush")
                        gr.HTML(PANO_FOOT, elem_classes="pg-bare")
                        examples = [[str(p)] for p in PANO_EXAMPLES if p.is_file()]
                        if examples:
                            head("Examples")
                            with gr.Column(elem_classes=["pg-examples", "pg-ex-pano"]):
                                gr.Examples(examples, inputs=v_in, label="", examples_per_page=8)
                        _, _, v_mode, v_explain = options(photo=False)
                        v_run = gr.Button("Find the view", elem_classes="pg-go")
                        v_json = gr.File(label="result.json", height=56, visible=False, elem_classes="pg-file")
                        with gr.Accordion("Progress log", open=False, elem_classes="pg-log"):
                            v_log = gr.Textbox(lines=10, max_lines=10, show_label=False, autoscroll=True)
                    with gr.Column(scale=7):
                        head("Chosen view on the panorama")
                        v_overlay = gr.Image(show_label=False, interactive=False, height=330, elem_classes="pg-media")
                        with gr.Row(equal_height=True, elem_classes="pg-eq"):
                            with gr.Column(scale=6):
                                head("Final frame")
                                v_final = gr.Image(show_label=False, interactive=False, height=290,
                                                   elem_classes="pg-media")
                            with gr.Column(scale=5):
                                head("Assessments")
                                v_scores = gr.HTML(EMPTY_SCORES, elem_classes=["pg-bare", "pg-fillh"])
                        head("Explanation")
                        v_text = gr.HTML(EMPTY_TEXT, elem_classes="pg-bare")
                v_run.click(frame_pano, [v_in, v_mode, v_explain],
                            [v_overlay, v_final, v_scores, v_text, v_json, v_log], concurrency_id="gpu")
    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    ap = argparse.ArgumentParser(description="PhotoGuider web GUI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true", help="also open a temporary public Gradio link")
    a = ap.parse_args()
    if not MAIN_PY.is_file():
        sys.exit(f"main environment not found at {MAIN_PY}; run envs/setup_envs.sh first")
    shutil.which("nvidia-smi") or print("nvidia-smi not found; GPU memory checks are skipped")
    build().launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
