"""Explaining a chosen crop: the four token assessments and the written paragraph.

    scores, tokens, overview = dashboard(model, frame, i_final, top1_box, final_box)
    geometry = mask_geometry(mask01, box, W, H)                  # measured subject geometry of the crop
    prompt, slot = build_prompt(scores, overview, geometry)      # the deployed prompt (anchors + silent bands)
    text = write_explanation(crop_path, prompt)                  # fine-tuned Qwen2.5-VL, in its own environment

Assessments: the final box is featurized on the same per-crop path the ranker uses, the ranker's
verdict is decomposed by TreeSHAP into the 60 feature attributions, those are summed per token
(S subject, C composition, A aesthetics, I integrity) and each sum is mapped to a percentile against
a fixed reference of near-human training crops (explain_data/dashboard_calibration.json). The
decomposition propagates each ranker member through the same per-image min-max as the selection;
the isotonically calibrated member is propagated by a secant (first order), the other three exactly.

Prompt rules (the paper's deployed configuration, unchanged): the four scores, the measured
geometry in photographic language, and deterministic anchor sentences for the subject's horizontal
position (always), its size share (always) and whether it is cut off (silent inside the ambiguous
band). Without a reliable subject mask the prompt forbids geometric claims entirely.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INF_DIR = HERE.parent
MODELS = INF_DIR / "models"
DATA = INF_DIR / "explain_data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INF_DIR))
from release_model import CORE_FEATURES, FC, with_pi, lgbm_quiet  # noqa: E402
from env_config import require_env  # noqa: E402

TOKENS = ["S", "C", "A", "I"]
TOKNAME = {"S": "Subject", "C": "Composition", "A": "Aesthetics", "I": "Integrity"}
# per-image min-max ensemble weights in collected form: 0.4 q_A + 0.4 q_B + 0.2 q_C with
# q_A = 0.6 rA + 0.4 kA, q_B = 0.55 rA + 0.45 kA, q_C = 0.6 rC(iso) + 0.4 kC
W_ENS = dict(rA=0.46, kA=0.34, rC=0.12, kC=0.08)


# ───────────────────────────── calibration tables ─────────────────────────────
def load_groups():
    g = json.load(open(DATA / "token_groups.json"))
    return {f: v["token"] for f, v in g["features"].items()}


def load_calibration():
    return json.load(open(DATA / "dashboard_calibration.json"))


# ───────────────────────────── TreeSHAP decomposition ─────────────────────────────
def pool_contribs(ranker, df_rows):
    """Per-feature TreeSHAP contributions (incl. bias) + raw member outputs for one image's rows."""
    Xf = df_rows[CORE_FEATURES].values.astype(np.float64)
    Xc = df_rows[FC].values.astype(np.float64)
    with lgbm_quiet():
        C = dict(rA=ranker["rA"].predict(Xf, pred_contrib=True), kA=ranker["kA"].predict(Xf, pred_contrib=True),
                 rC=ranker["rC"].predict(Xc, pred_contrib=True), kC=ranker["kC"].predict(Xc, pred_contrib=True))
    raw = {m: C[m].sum(axis=1) for m in C}
    raw["rC_iso"] = ranker["iso"].predict(raw["rC"])
    return C, raw


def base_decomp(i, C, raw):
    """Per-feature contribution of row i, propagated through the ensemble's per-image min-max slopes.
    Sign retained. Exact for rA/kA/kC; secant (first order) through the isotonic head of rC."""
    out = {f: 0.0 for f in FC}
    for m, F in (("rA", CORE_FEATURES), ("kA", CORE_FEATURES), ("kC", FC)):
        r = raw[m]; lo, hi = float(r.min()), float(r.max())
        if hi <= lo:
            continue
        s = W_ENS[m] / (hi - lo)
        for fi, f in enumerate(F):
            out[f] += s * C[m][i, fi]
    r, ri = raw["rC"], raw["rC_iso"]
    lo, hi = float(ri.min()), float(ri.max())
    if hi > lo and r[i] != r.mean():
        slope = (ri[i] - ri.mean()) / (r[i] - r.mean())
        s = W_ENS["rC"] * slope / (hi - lo)
        for fi, f in enumerate(FC):
            out[f] += s * C["rC"][i, fi]
    return out


def gauge(table, token, value):
    """Signed token contribution -> 0-100 percentile against the reference distribution."""
    arr = np.asarray(table["tokens"][token]["values_sorted"], dtype=np.float64)
    n = arr.size
    lo = np.searchsorted(arr, value, side="left")
    hi = np.searchsorted(arr, value, side="right")
    return int(round(100.0 * (lo + hi) / (2.0 * n)))


def _adj(v):
    return "stands out" if v >= 70 else ("is solid" if v >= 40 else "is soft")


def overview(scores, moved_px, scale_action):
    """One short English sentence: the two strongest gauges + the refinement action taken."""
    top = sorted(TOKENS, key=lambda t: -scores[t])[:2]
    acts = []
    if moved_px > 1:
        acts.append("nudged the framing")
    if scale_action == "expand":
        acts.append("zoomed out slightly")
    elif scale_action == "shrink":
        acts.append("tightened the frame")
    act = ("the system " + " and ".join(acts) + ".") if acts else "the framing was kept as-is."
    return (f"The {TOKNAME[top[0]].lower()} {_adj(scores[top[0]])} ({scores[top[0]]}), "
            f"the {TOKNAME[top[1]].lower()} {_adj(scores[top[1]])} ({scores[top[1]]}); {act}")


def featurize_final(pipeline, extractors, k10_boxes, final_box, pid_key, image_path, W, H, vmix, v1, mask):
    """K10 boxes + the final box as one frame, featurized on the same per-crop path the pipeline uses
    (the K10 rows give the per-image normalization its context)."""
    rows = []
    for k, b in enumerate(list(k10_boxes) + [list(final_box)]):
        asp = (b[2] - b[0]) / max(b[3] - b[1], 1e-6)
        rows.append(dict(photo_id=pid_key, image_path=str(image_path), src="k10" if k < len(k10_boxes) else "final",
                         x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                         img_w=float(W), img_h=float(H), gt_aspect=float(asp), dpp_rank=k))
    man = pd.DataFrame(rows)
    feats = pipeline.af_chain_features(man, vmix, v1, mask)
    frame = pd.concat([man.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = extractors.deep(frame, subject_masks={pid_key: (None, None, mask)})
    frame = with_pi(frame)
    # stage_c_features reports inspy_capture/inspy_max on the 0-255 scale; dividing by 255 puts them on the
    # 0-1 scale of the conditioned pool (pool_sources) and of the FCDB rows the ranker was trained on. For
    # free-form images the decision itself used the 0-255 values (FLMS/GAICD convention), so this
    # attribution frame differs from the decision frame in these two columns; the shipped explanation
    # records and the paper's faithfulness numbers use this frame.
    frame["inspy_capture"] = frame["inspy_capture"] / 255.0
    frame["inspy_max"] = frame["inspy_max"] / 255.0
    return frame


def dashboard(ranker, feat2tok, table, frame, i_final, top1_box, final_box):
    """(scores, token_values, overview_sentence) for the final box (row ``i_final`` of ``frame``)."""
    C, raw = pool_contribs(ranker, frame)
    dec = base_decomp(i_final, C, raw)
    tok = {t: 0.0 for t in TOKENS}
    for f, v in dec.items():
        tok[feat2tok[f]] += v
    scores = {t: gauge(table, t, tok[t]) for t in TOKENS}
    cx = ((top1_box[0] + top1_box[2]) / 2, (top1_box[1] + top1_box[3]) / 2)
    fx = ((final_box[0] + final_box[2]) / 2, (final_box[1] + final_box[3]) / 2)
    moved = float(np.hypot(fx[0] - cx[0], fx[1] - cx[1]))
    a0 = max((top1_box[2] - top1_box[0]) * (top1_box[3] - top1_box[1]), 1.0)
    a1 = max((final_box[2] - final_box[0]) * (final_box[3] - final_box[1]), 1.0)
    sr = float(np.sqrt(a1 / a0))
    action = "expand" if sr > 1.02 else ("shrink" if sr < 0.98 else "keep")
    return scores, tok, overview(scores, moved, action)


# ───────────────────────────── measured geometry of the crop ─────────────────────────────
SYM_T = 0.05            # sym_h_diff below -> horizontally symmetric
THIRDS_ON, THIRDS_OFF = 0.08, 0.15
CENTER_LO, CENTER_HI = 0.40, 0.60
CUT_IN, CUT_OUT = 0.03, 0.12                  # inside band: entirely inside; outside: cut; between: silent
SIZE_BANDS = ((0.10, "sits small in"), (0.30, "occupies a moderate share of"), (1.01, "dominates"))
NEAR_TIE = 5


def mask_geometry(mask01, box, W, H):
    """Subject-mask geometry inside the crop box (mask in [0,1], any resolution). See OUTPUT_SCHEMA."""
    g = dict(mask_ok=False)
    if mask01 is None:
        return g
    mk = mask01.astype(np.float32)
    hp, wp = mk.shape
    total = float(mk.sum())
    x1, y1, x2, y2 = box
    px1 = max(0, min(int(x1 / W * wp), wp - 1)); py1 = max(0, min(int(y1 / H * hp), hp - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * wp)), wp))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * hp)), hp))
    inner = mk[py1:py2, px1:px2]
    if total / (hp * wp) > 0.005 and inner.size > 4:
        cap = float(inner.sum() / (total + 1e-9))
        im = float(inner.sum())
        if im > 1e-6:
            yy, xx = np.indices(inner.shape)
            g = dict(mask_ok=True, capture=cap, lost=1.0 - cap,
                     cx=float((inner * xx).sum() / im) / max(inner.shape[1] - 1, 1),
                     cy=float((inner * yy).sum() / im) / max(inner.shape[0] - 1, 1),
                     ssr=float(inner.mean()))
    return g


# ───────────────────────────── the deployed prompt ─────────────────────────────
PROMPT = (
    "This photo crop was selected by an automatic composition system. "
    "Calibrated indicator scores (0-100, higher=better, percentile vs human-approved crops): "
    "Subject={S}, Composition={C}, Aesthetics={A}, Integrity={I}. "
    "Indicator definitions: Subject = subject placement and prominence; "
    "Composition = geometric arrangement (thirds, balance, lines); "
    "Aesthetics = visual appeal of the arrangement; "
    "Integrity = completeness of scene elements (nothing important cut off). "
    "{overview}"
    "{facts} "
    "In at most 3 concise sentences, explain why this crop works, referencing ONLY things "
    "visible in the image and the given scores. "
    "{naming} "
    "Discuss ONLY composition: framing, subject placement, balance, leading lines, space, "
    "and whether elements are cut off. Do NOT comment on sharpness, blur, focus, clarity, "
    "noise, resolution, exposure, or any technical image quality. "
    "Do not mention any object, color, or person you cannot clearly see. Do not invent numbers."
)
POS_TEXT = {"left": "in the left third of the frame", "center": "horizontally centered",
            "right": "in the right third of the frame"}
GEO_INSTR = (" For position, cut-off, symmetry, rule-of-thirds alignment, and size, state "
             "ONLY what the measured geometry says — do not judge these from the image. "
             "Your task: identify what the subject is and connect the measured facts and "
             "scores into 2-3 fluent sentences about the composition.")
NOMASK_TEXT = "No reliable geometric measurements are available for this image."
NOMASK_INSTR = (" IMPORTANT: because no geometric measurements are available, do NOT "
                "mention subject position, centering, placement in the frame, anything "
                "being cut off or cropped, symmetry, the rule of thirds, or subject size "
                "at all. Discuss ONLY the indicator scores and what the subject visibly is. "
                "When explaining the Integrity score, speak of overall scene completeness "
                "in general terms and never use phrases like 'cut off' or 'cropped'; when "
                "explaining the Subject score, speak of prominence and visibility, never "
                "of placement or position.")
NOMASK_FINAL = (" FINAL OVERRIDING RULE for this image: no geometric measurements exist, "
                "so do NOT mention subject position, placement, centering, anything being "
                "cut off or cropped, symmetry, the rule of thirds, or subject size — even "
                "though the general instructions above list such topics. You may say a "
                "score is high or low, but never justify it with a geometric statement. "
                "Talk only about what the subject is and the given indicator scores, in "
                "English only.")
SINGLE_STATEMENT = (
    " State each compositional aspect only once: do not give two different or opposing "
    "judgements of the same aspect anywhere in your answer."
)


def rank_indicators(scores):
    order = sorted("SCAI", key=lambda k: (-scores[k], k))
    s_tie = scores[order[0]] - scores[order[1]] < NEAR_TIE
    w_tie = scores[order[-2]] - scores[order[-1]] < NEAR_TIE
    return ([order[0], order[1]] if s_tie else [order[0]], None if w_tie else order[-1])


def facts_text(scores, strongest, weakest):
    if len(strongest) == 2:
        f = (f"GIVEN FACTS (already computed, do not re-derive or dispute): the strongest "
             f"indicators are {TOKNAME[strongest[0]]} ({scores[strongest[0]]}) and "
             f"{TOKNAME[strongest[1]]} ({scores[strongest[1]]}), nearly tied")
    else:
        f = (f"GIVEN FACTS (already computed, do not re-derive or dispute): the strongest "
             f"indicator is {TOKNAME[strongest[0]]} ({scores[strongest[0]]})")
    if weakest is None:
        f += ". The weakest indicators are nearly tied, so no weakest is designated."
    else:
        f += f"; the weakest is {TOKNAME[weakest]} ({scores[weakest]})."
    return f


def build_slot(strongest, weakest):
    s = "Strongest: " + " and ".join(TOKNAME[k] for k in strongest) + "."
    if weakest is not None:
        s += f" Weakest: {TOKNAME[weakest]}."
    return s


def slot_instr(slot, weakest):
    n = f'Your response MUST begin with exactly this first sentence, copied verbatim: "{slot}"'
    if weakest is None:
        n += " Do not name or discuss a weakest indicator after that."
    n += " After this required opening sentence, continue with the explanation."
    return n


def thirds_zone(cx, cy):
    dx = min(abs(cx - 1 / 3), abs(cx - 2 / 3))
    dy = min(abs(cy - 1 / 3), abs(cy - 2 / 3))
    if min(dx, dy) < THIRDS_ON:
        return "on"
    if dx > THIRDS_OFF and dy > THIRDS_OFF:
        return "off"
    return "band"


def geo_text(g):
    if not g.get("mask_ok"):
        return NOMASK_TEXT
    pos = "left" if g["cx"] < CENTER_LO else ("center" if g["cx"] <= CENTER_HI else "right")
    parts = [f"the subject is {POS_TEXT[pos]} "
             f"(horizontal center at {g['cx']:.0%} of the frame width)"]
    if g["capture"] >= 0.95:
        parts.append("the subject is fully contained in the frame")
    else:
        parts.append(f"the subject is partially cut off "
                     f"(~{max(g['lost'], 0.0):.0%} of the subject outside the frame)")
    parts.append(f"the subject occupies ~{g['ssr']:.0%} of the frame")
    parts.append("the crop is horizontally symmetric" if g["sym_h_diff"] < SYM_T
                 else "the crop is not horizontally symmetric")
    tz = thirds_zone(g["cx"], g["cy"])
    if tz == "on":
        parts.append("the subject aligns well with the rule of thirds")
    elif tz == "off":
        parts.append("the subject is not aligned with the rule of thirds")
    return "Measured geometry (authoritative): " + "; ".join(parts) + "."


def anchor_lines(g):
    """Direction (always), size (always), cut-off (silent in its ambiguous band)."""
    if not g.get("mask_ok"):
        return []
    lines = []
    cx = g["cx"]
    bucket = ("left third" if cx < CENTER_LO else ("centre" if cx <= CENTER_HI else "right third"))
    lines.append(f"The subject sits in the {bucket}, about {round(cx * 100)}% across the frame.")
    if g.get("ssr") is not None:
        word = next((w for hi, w in SIZE_BANDS if g["ssr"] < hi), SIZE_BANDS[-1][1])
        lines.append(f"The subject {word} the frame, filling about {round(g['ssr'] * 100)}% of it.")
    lost = g.get("lost")
    if lost is not None:
        if lost > CUT_OUT:
            lines.append("Parts of the subject fall outside the frame.")
        elif lost <= CUT_IN:
            lines.append("The subject is entirely inside the frame.")
    return lines


def build_prompt(scores, overview_sentence, geometry):
    strongest, weakest = rank_indicators(scores)
    slot = build_slot(strongest, weakest)
    facts = facts_text(scores, strongest, weakest)
    facts += " " + geo_text(geometry) + (GEO_INSTR if geometry.get("mask_ok") else NOMASK_INSTR)
    ov = f"System overview: '{overview_sentence}'. " if overview_sentence else ""
    prompt = PROMPT.format(S=scores["S"], C=scores["C"], A=scores["A"], I=scores["I"],
                           overview=ov, facts=facts, naming=slot_instr(slot, weakest))
    lines = anchor_lines(geometry)
    if lines:
        prompt = prompt.rstrip() + " " + " ".join(lines)
    if not geometry.get("mask_ok"):
        prompt += NOMASK_FINAL              # must precede the closing rule below
    prompt += SINGLE_STATEMENT
    return prompt, slot


# ───────────────────────────── the explanation VLM ─────────────────────────────
ADAPTER_DIR = MODELS / "explainer"


def write_explanation(crop_path, prompt):
    """One greedy decode of the fine-tuned explainer, run in the configured explainer environment."""
    request = {"image": str(Path(crop_path).resolve()), "prompt": prompt, "adapter_dir": str(ADAPTER_DIR)}
    proc = subprocess.run([str(require_env("explainer")), str(HERE / "explain_vlm_worker.py")],
                          input=json.dumps(request), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"explainer worker failed ({proc.returncode}):\n{proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)["explanation"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"invalid explainer worker response: {proc.stdout!r}") from exc


def load_ranker():
    return {n: joblib.load(MODELS / f"ranker_{n}.pkl") for n in ["rA", "kA", "rC", "kC", "iso"]}
