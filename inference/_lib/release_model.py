"""The released PhotoGuider decision layer, as ONE deep module.

Everything numerical that turns a featurized candidate frame into a final crop lives here, in one
copy, loaded from the shipped artifacts in ``inference/models/``:

    model = ReleaseModel.load(models_dir)
    q     = model.score(frame)                       # 60-feature ranker ensemble, per-image normalized
    rc    = model.recenter(crop, feats, thirds, cam, W, H)
    box   = model.scale(rc, protocol, ...)           # four expansion gates (MAX-union) + shrink fallback
    v     = consensus_box(greedy, samples)           # consultant opinions
    box   = blend(box, v, protocol, W, H)            # agreement-gated concession

Function bodies are the validated benchmark algorithms, moved here verbatim from the former
reproduce.py / infer.py pair of copies that used to drift. The benchmark
(reproduce.py) and the single-image path (infer.py) both call this module, so a change here is a
change everywhere, and the golden test (tests/test_reproduce_golden.py) pins the numbers.

Interface facts a caller must know:
  * ``frame`` is a pandas DataFrame with the 60 columns of ``FC`` plus ``photo_id``; ``score``
    normalizes per ``photo_id`` (min-max over the rows of each image), so the rows of one image
    must be scored together — scoring a single row is meaningless.
  * ``laion_pi_minmax`` is a per-image derived column; recompute it with ``per_img_minmax`` over
    the exact candidate set being scored (see ``with_pi``).
  * Boxes are ``(x1, y1, x2, y2)`` in pixels of the image whose size is ``(W, H)``; every
    geometric move clips to the image and preserves the box size where it can.
  * The ranker picks among DPP ranks ``< Protocol.k_sel`` (3); twin consultant rows carry
    ``dpp_rank >= 10`` and are compared through ``twin_flip``.
"""
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ───────────────────────────── feature vocabulary (60) ─────────────────────────────
# 17 core saliency/geometry features (the "core compositional set" of the paper)
CORE_FEATURES = ["vmix_max", "v1_max", "v1_cov", "inspy_capture", "inspy_max", "ca_dist", "area_ratio",
                 "log_aspect_diff", "thirds_align", "sym_v_diff", "sym_diag_diff", "tri_top_apex",
                 "tri_bot_apex", "color_sat_mean", "visual_weight_x", "visual_weight_y", "subject_size_ratio"]
LAION = ["laion_score", "laion_pi_minmax"]
SAM_ENC = ["sam_emb_norm", "sam_emb_var", "sam_emb_kurt", "sam_cos_to_image", "sam_cand_norm_mean",
           "sam_cand_norm_max", "sam_cand_norm_std", "sam_cand_to_outside_ratio", "sam_image_total_norm",
           "sam_mask_area_ratio", "sam_mask_centered_x", "sam_mask_centered_y", "sam_mask_vs_inspy_iou"]
SAM_DEC = ["sam_mask_q_mean", "sam_mask_q_std", "sam_mask_q_margin", "sam_h_area_cv", "sam_h_iou_top12",
           "sam_h_cand_match_max", "sam_obj_count_in_cand", "sam_obj_top1_share", "sam_obj_top3_share",
           "sam_obj_area_cv", "sam_dominant_ratio", "sam_fg_bg_sat_diff", "sam_fg_bg_val_diff",
           "sam_fg_bg_lab_dist", "sam_fg_var", "sam_thirds_q_mean", "sam_thirds_q_max", "sam_thirds_vs_center"]
FG = ["fgaesq_score"]
CMP = ["clip_sim_pro", "clip_sim_balanced", "clip_sim_leading", "clip_sim_thirds", "clip_sim_symmetric",
       "clip_sim_negspace", "clip_sim_centered", "clip_sim_dynamic", "clip_max_sim"]
FC = CORE_FEATURES + LAION + SAM_ENC + SAM_DEC + FG + CMP
assert len(FC) == 60

# refinement constants
BETA = 0.20            # recenter: fraction of the way toward the target
THIRDS_W = 0.4         # thirds target: weight of the snap toward the nearest thirds point
GATE_T = 0.05          # recenter gate: predicted CLIP-target advantage that maps to w = 1
SS = [1.1, 1.3, 1.5, 2.0]        # expansion factors
SS_SHR = [0.6, 0.7, 0.8, 0.9]    # shrink factors
PK = ["pro", "balanced", "leading", "thirds", "symmetric", "negspace", "centered", "dynamic", "aesthetic"]
COMP = ["pro", "balanced", "leading", "thirds", "symmetric", "centered"]
GATE_EXTRA = ["sam_obj_count_in_cand", "sam_dominant_ratio", "sam_obj_top1_share", "thirds_align", "inspy_capture", "vmix_max"]

# consultant constants
TWIN_DELTA = 0.3       # isolated-flip margin a twin must beat the clean winner by
TWIN_BC, TWIN_BA, TWIN_IOU = 0.4, 0.2, 0.5   # twin construction: center / area blend, agreement gate
DEGEN = 0.90           # prompt-ensemble boxes covering >= 90% of the image are discarded
RANKER_FILES = ("rA", "kA", "rC", "kC", "iso")
GATE_FILES = ("pan", "cov", "comp", "gold", "aes", "shrink")


# ───────────────────────────── the two protocols ─────────────────────────────
@dataclass(frozen=True)
class Protocol:
    """Mode constants. ``conditioned`` = a target aspect is requested (FCDB protocol);
    ``free_form`` = no aspect input (FLMS protocol)."""
    name: str
    st: float      # expansion-gate threshold on predicted IoU gain
    sh: float      # shrink-gate threshold
    bc: float      # blend: center lerp toward the consultant
    ba: float      # blend: geometric-mean area toward the consultant
    tau: float     # blend: agreement IoU required before any concession
    br: float      # blend: aspect lerp (0 keeps the frame's ratio, 1 adopts the consultant's)
    k_sel: int = 3           # ranker selects among DPP ranks < k_sel
    twin_delta: float = TWIN_DELTA
    degen: float = DEGEN


CONDITIONED = Protocol("conditioned", st=0.03, sh=0.05, bc=0.4, ba=0.2, tau=0.5, br=0.0)
FREE_FORM = Protocol("free_form", st=-0.02, sh=0.08, bc=0.7, ba=0.3, tau=0.45, br=1.0)


def protocol_for(target_aspect) -> Protocol:
    return CONDITIONED if target_aspect is not None else FREE_FORM


# ───────────────────────────── per-image normalization ─────────────────────────────
def per_img_minmax(df, col):
    o = np.zeros(len(df))
    for pid, idx in df.groupby("photo_id").groups.items():
        v = df.loc[idx, col].values; lo, hi = float(v.min()), float(v.max())
        o[idx] = (v - lo) / (hi - lo) if hi > lo else 0.5
    return o


def with_pi(df):
    """Recompute ``laion_pi_minmax`` over exactly these rows and check the 60 features are present and finite."""
    d = df.reset_index(drop=True).copy()
    d["laion_pi_minmax"] = per_img_minmax(d, "laion_score")
    missing = sorted(set(FC) - set(d.columns))
    if missing or not np.isfinite(d[FC].to_numpy(dtype=float)).all():
        raise ValueError(f"ranker requires 60 finite features per candidate; missing columns: {missing}")
    return d


def ens(rp, kp, df, wr, wk, iso=None):
    d = df.copy(); qr = rp.copy()
    if iso is not None: qr = iso.predict(qr)
    d["qr"] = qr; d["qk"] = kp; d["qrn"] = per_img_minmax(d, "qr"); d["qkn"] = per_img_minmax(d, "qk")
    return (wr * d.qrn + wk * d.qkn).values


# ───────────────────────────── geometry ─────────────────────────────
def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    I = (ix2 - ix1) * (iy2 - iy1); U = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I
    return max(0.0, min(1.0, I / U)) if U > 0 else 0.0


def mi(crop, G, multi):
    """IoU against one GT box, or the max over several (FLMS has multiple GT boxes)."""
    return max(iou_xyxy(crop, b) for b in G) if multi else iou_xyxy(crop, G)


@contextlib.contextmanager
def lgbm_quiet():
    """The LightGBM models are called on plain arrays; silence only their feature-name notice, only here."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


def _clip_box(x1, y1, x2, y2, W, H):
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > W: x1 -= (x2-W); x2 = W
    if y2 > H: y1 -= (y2-H); y2 = H
    return (max(0, x1), max(0, y1), min(W, x2), min(H, y2))


def grid_cent(g):
    g = g - g.min()
    if g.sum() <= 0: return None
    yy, xx = np.indices(g.shape); t = g.sum()
    return float((g*xx).sum()/t/16 + 0.5/16), float((g*yy).sum()/t/16 + 0.5/16)


def salient_center(mask, thr=0.3):
    if mask is None or mask.max() <= 0: return None
    mh, mw = mask.shape; ys, xs = np.where(mask >= thr * mask.max())
    if len(xs) == 0: return None
    return ((xs.min() + xs.max()) / 2) / mw, ((ys.min() + ys.max()) / 2) / mh


def thirds_snap(p, w=THIRDS_W):
    if p is None: return None
    tx = min([1/3, 2/3], key=lambda t: abs(t - p[0])); ty = min([1/3, 2/3], key=lambda t: abs(t - p[1]))
    return (1-w)*p[0] + w*tx, (1-w)*p[1] + w*ty


def cam_center(heat):
    if heat is None: return None
    return grid_cent(heat[:, :, [PK.index(p) for p in COMP]].mean(2))


def recenter_to(crop, tgt, beta, W, H):
    if tgt is None: return crop
    cx, cy = (crop[0]+crop[2])/2, (crop[1]+crop[3])/2; w, h = crop[2]-crop[0], crop[3]-crop[1]
    ncx = (1-beta)*cx + beta*tgt[0]*W; ncy = (1-beta)*cy + beta*tgt[1]*H
    return _clip_box(ncx-w/2, ncy-h/2, ncx+w/2, ncy+h/2, W, H)


def blend_tgt(th, cam, w):
    if th is None: return cam
    if cam is None: return th
    return ((1-w)*th[0] + w*cam[0], (1-w)*th[1] + w*cam[1])


def expand_s(crop, s, W, H):
    cx, cy = (crop[0]+crop[2])/2, (crop[1]+crop[3])/2; w = (crop[2]-crop[0])*s; h = (crop[3]-crop[1])*s
    return _clip_box(cx-w/2, cy-h/2, cx+w/2, cy+h/2, W, H)


def capture(mask, crop, W, H, thr=0.3):
    if mask is None or mask.max() <= 0: return 0.0, 0.0
    m = np.asarray(mask).astype(np.float32); h, w = m.shape
    b = (m >= thr * m.max()).astype(np.float32); tot = b.sum()
    if tot <= 0: return 0.0, 0.0
    x1, y1, x2, y2 = crop
    xs = np.clip(np.minimum(np.arange(1, w+1), x2/W*w) - np.maximum(np.arange(0, w), x1/W*w), 0, 1)
    ys = np.clip(np.minimum(np.arange(1, h+1), y2/H*h) - np.maximum(np.arange(0, h), y1/H*h), 0, 1)
    wgt = np.outer(ys, xs); inside = (b * wgt).sum(); area = wgt.sum()
    return float(inside/tot), float(inside/max(area, 1e-6))


# ───────────────────────────── gate feature vectors ─────────────────────────────
def gate_feats(t, th, cam):
    """Recenter-gate input for candidate row ``t`` (needs img_w/h, box, subject_size_ratio, area_ratio,
    sam_mask_area_ratio and GATE_EXTRA)."""
    W, H = float(t.img_w), float(t.img_h)
    cx = (t.x1+t.x2)/2/W; cy = (t.y1+t.y2)/2/H; cw = (t.x2-t.x1)/W; ch = (t.y2-t.y1)/H
    sb_dx = (th[0]-cx) if th else 0.0; sb_dy = (th[1]-cy) if th else 0.0
    cam_dx = (cam[0]-cx) if cam else 0.0; cam_dy = (cam[1]-cy) if cam else 0.0
    gap = (np.hypot(th[0]-cam[0], th[1]-cam[1]) if (th and cam) else 0.0)
    ssr = float(t.subject_size_ratio)
    rf = [cx, cy, cw, ch, sb_dx, sb_dy, ssr, sb_dx, sb_dy, cam_dx, cam_dy, gap,
          ssr, float(t.area_ratio), float(t.sam_mask_area_ratio)]
    return rf + [float(t[c]) for c in GATE_EXTRA]


def sfeat(crop, W, H, mask, ssr, s):
    cap, den = capture(mask, crop, W, H); cw = (crop[2]-crop[0])/W; ch = (crop[3]-crop[1])/H
    return [cap, den, cw*ch, np.log((cw+1e-6)/(ch+1e-6)), ssr, s]


def ctf(ct, pid):
    v = ct.get(pid)
    if v is None: return 0.0, 0.0, 0.0
    wide = float(v[:6].mean()); tight = float(v[6:10].mean())
    return wide - tight, wide, tight


def sfeat_ws(crop, W, H, ssr, s, ct, pid):
    cw = (crop[2]-crop[0])/W; ch = (crop[3]-crop[1])/H
    ws, wi, ti = ctf(ct, pid)
    return [ws, wi, ti, cw*ch, np.log((cw+1e-6)/(ch+1e-6)), ssr, s]


def sfeat_aes(crop, W, H, ssr, s, alut, grp, pid):
    cw = (crop[2]-crop[0])/W; ch = (crop[3]-crop[1])/H
    dv = alut.get((grp, str(pid), s), 0.0)
    return [float(dv), cw*ch, np.log((cw+1e-6)/(ch+1e-6)), ssr, s]


def sfeat_gold(crop, W, H, ssr, s, glut, grp, pid):
    cw = (crop[2]-crop[0])/W; ch = (crop[3]-crop[1])/H
    dv = glut.get((grp, str(pid), s), 0.0)
    return [float(dv), cw*ch, np.log((cw+1e-6)/(ch+1e-6)), ssr, s]


def sfeat_shrink(crop, W, H, mask, ssr, s, samcos):
    cap, den = capture(mask, crop, W, H); cw = (crop[2]-crop[0])/W; ch = (crop[3]-crop[1])/H
    return [float(samcos), cap, den, cw*ch, np.log((cw+1e-6)/(ch+1e-6)), ssr, s]


# ───────────────────────────── consultant opinions + blends ─────────────────────────────
def consensus_box(greedy, slots):
    """Coordinate-wise median of the greedy box and the valid sampled boxes; the greedy box alone if none."""
    boxes = [tuple(greedy)] + [tuple(s) for s in slots if s is not None]
    arr = np.array(boxes, dtype=np.float64)
    return tuple(np.median(arr, axis=0))


def expanded_opinion(greedy, multibox, promptens, W, H, degen=DEGEN):
    """The broader opinion v_f: greedy + temperature samples + prompt-ensemble boxes, the latter
    filtered of near-whole-image answers (area >= degen * W * H)."""
    ps = [b for b in (promptens or []) if b is not None and max(b[2]-b[0], 0)*max(b[3]-b[1], 0) < degen*W*H]
    return consensus_box(greedy, list(multibox or []) + ps)


def blend_box(fc, V, bc, ba, W, H):
    """Conditioned blend: center lerp ``bc`` + geometric-mean area ``ba``; aspect preserved."""
    if bc == 0.0 and ba == 0.0: return fc
    cx, cy = (fc[0]+fc[2])/2, (fc[1]+fc[3])/2; vcx, vcy = (V[0]+V[2])/2, (V[1]+V[3])/2
    ncx = (1-bc)*cx + bc*vcx; ncy = (1-bc)*cy + bc*vcy
    ao = max((fc[2]-fc[0])*(fc[3]-fc[1]), 1.0); av = max((V[2]-V[0])*(V[3]-V[1]), 1.0)
    na = (ao**(1-ba)) * (av**ba); asp = (fc[2]-fc[0]) / max(fc[3]-fc[1], 1e-6)
    nw = min(math.sqrt(na*asp), W); nh = min(math.sqrt(na/asp), H)
    return _clip_box(ncx-nw/2, ncy-nh/2, ncx+nw/2, ncy+nh/2, W, H)


def blend_box_a(fc, V, bc, ba, br, W, H):
    """Free-form blend: as ``blend_box`` plus a log-space aspect lerp ``br`` toward the consultant's ratio."""
    cx, cy = (fc[0]+fc[2])/2, (fc[1]+fc[3])/2; vcx, vcy = (V[0]+V[2])/2, (V[1]+V[3])/2
    ncx = (1-bc)*cx + bc*vcx; ncy = (1-bc)*cy + bc*vcy
    ao = max((fc[2]-fc[0])*(fc[3]-fc[1]), 1.0); av = max((V[2]-V[0])*(V[3]-V[1]), 1.0)
    na = (ao**(1-ba)) * (av**ba)
    aspo = (fc[2]-fc[0]) / max(fc[3]-fc[1], 1e-6); aspv = (V[2]-V[0]) / max(V[3]-V[1], 1e-6)
    asp = math.exp((1-br)*math.log(aspo) + br*math.log(max(aspv, 1e-6)))
    nw = min(math.sqrt(na*asp), W); nh = min(math.sqrt(na/asp), H)
    return _clip_box(ncx-nw/2, ncy-nh/2, ncx+nw/2, ncy+nh/2, W, H)


def blend(fc, V, protocol: Protocol, W, H):
    """Agreement-gated concession: blend only when IoU(frame, opinion) >= tau. Returns (box, blended)."""
    if V is None or iou_xyxy(fc, V) < protocol.tau:
        return fc, False
    if protocol.br > 0:
        return blend_box_a(fc, V, protocol.bc, protocol.ba, protocol.br, W, H), True
    return blend_box(fc, V, protocol.bc, protocol.ba, W, H), True


def make_twin(cand, v6, W, H):
    """A twin candidate: ``cand`` moved toward the core opinion ``v6`` within the fixed twin blend."""
    return blend_box(cand, v6, TWIN_BC, TWIN_BA, W, H)


def twin_flip(score_fn, delta):
    """Wrap ``score_fn`` so that, per image, the best twin row (dpp_rank >= 10) wins only if its score
    on the merged pool beats the clean K-pool winner by ``delta``; images without twins are untouched."""
    def q(df):
        kmask = (df.dpp_rank.values < 10)
        out = np.full(len(df), -1e9)
        df10 = df[kmask].reset_index(drop=True)
        out[kmask] = score_fn(df10)                       # clean K10 q (exact baseline ordering)
        q11 = score_fn(df)                                # contaminated; compare-only
        d = pd.DataFrame({"pid": df.photo_id.values, "k": kmask, "q10": out, "q11": q11})
        for pid, s in d.groupby("pid", sort=False):
            tw = s[~s.k]
            if not len(tw):                          # no twins for this pid
                continue
            wi = s[s.k].q10.idxmax()                 # clean winner row (position)
            vi = tw.q11.idxmax()                     # best twin by q11
            if d.q11.loc[vi] > d.q11.loc[wi] + delta:
                out[vi] = 1e9                        # force twin pick
        return out
    return q


# ───────────────────────────── the model ─────────────────────────────
class ReleaseModel:
    """The shipped decision layer: four ranker members + isotonic calibrator + six refinement gates."""

    def __init__(self, ranker: dict, gates: dict):
        self.ranker = ranker
        self.gates = gates

    @classmethod
    def load(cls, models_dir) -> "ReleaseModel":
        d = Path(models_dir)
        missing = [n for n in (*[f"ranker_{r}" for r in RANKER_FILES], *[f"gate_{g}" for g in GATE_FILES])
                   if not (d / f"{n}.pkl").is_file()]
        if missing:
            raise FileNotFoundError(f"missing model artifacts in {d}: {missing}")
        return cls({n: joblib.load(d / f"ranker_{n}.pkl") for n in RANKER_FILES},
                   {n: joblib.load(d / f"gate_{n}.pkl") for n in GATE_FILES})

    # ranking
    def score(self, df):
        """q = 0.4 q_A + 0.4 q_B + 0.2 q_C, each member per-image min-max normalized over ``df``."""
        R = self.ranker
        xa, xc = df[CORE_FEATURES].values, df[FC].values
        with lgbm_quiet():
            qA = ens(R["rA"].predict(xa), R["kA"].predict(xa), df, 0.6, 0.4)
            qB = ens(R["rA"].predict(xa), R["kA"].predict(xa), df, 0.55, 0.45)
            qC = ens(R["rC"].predict(xc), R["kC"].predict(xc), df, 0.6, 0.4, iso=R["iso"])
        return 0.4 * qA + 0.4 * qB + 0.2 * qC

    def score_with_twins(self, df, delta=TWIN_DELTA):
        return twin_flip(self.score, delta)(df)

    # refinement
    def recenter_weight(self, gf):
        """w in [0,1]: how far the recenter target moves from the thirds point toward the CLIP target."""
        with lgbm_quiet():
            return float(np.clip(self.gates["pan"].predict([gf])[0] / GATE_T, 0, 1))

    def recenter(self, crop, gf, th, cam, W, H):
        return recenter_to(crop, blend_tgt(th, cam, self.recenter_weight(gf)), BETA, W, H)

    def scale_box(self, rc, W, H, protocol: Protocol, mask, ssr, samcos, ct, alut, glut, grp, pid):
        """Expansion gates (MAX-union over coverage / composition / golden / aesthetics) and, if none
        fires, the shrink fallback.
        Returns (box, scale_factor_or_None, best_expansion_prediction)."""
        G = self.gates
        with lgbm_quiet():
            pr = []
            for s in SS:
                cand = {"cov": G["cov"].predict([sfeat(rc, W, H, mask, ssr, s)])[0],
                        "comp": G["comp"].predict([sfeat_ws(rc, W, H, ssr, s, ct, pid)])[0],
                        "gold": G["gold"].predict([sfeat_gold(rc, W, H, ssr, s, glut, grp, pid)])[0],
                        "aes": G["aes"].predict([sfeat_aes(rc, W, H, ssr, s, alut, grp, pid)])[0]}
                pr.append(max(cand.values()))
            bi = int(np.argmax(pr))
            if pr[bi] > protocol.st:
                return expand_s(rc, SS[bi], W, H), SS[bi], float(pr[bi])
            ps = [G["shrink"].predict([sfeat_shrink(rc, W, H, mask, ssr, s, samcos)])[0] for s in SS_SHR]
        bs = int(np.argmax(ps))
        if ps[bs] > protocol.sh:
            return expand_s(rc, SS_SHR[bs], W, H), SS_SHR[bs], float(pr[bi])
        return rc, None, float(pr[bi])
