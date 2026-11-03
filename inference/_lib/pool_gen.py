"""Faithful single-image reproduction of the production candidate pool (8 sources).

Pool composition (~300 candidates/image):
  base geometry (center-shift grid + small-area grid + seeded random boxes)
    -> C          : each base candidate re-aspected to the requested ratio (NMS@0.95)
  subject-mask anchors -> INSPY_anc : centroid + blobs + mask-bbox x medium areas
  saliency top-5 peaks -> VMIX_peak : peaks x medium areas
  {C, INSPY, VMIX} top-20 by saliency coverage x {0.7, 0.85, 1.15, 1.3} -> multiscale
  composition-principle generators -> S_diagonal / S_negspace / S_leading / S_boundary

Reproducibility (verified on benchmark images): fed the same base geometry, this generator
reproduces the cached candidate pool byte-exactly across the deterministic sources. The `rand`
source uses a fixed per-image seed (RAND_SEED), so the pool is deterministic for any single image.
The S_* sources also get per-source NMS@0.95.
"""
import math
import numpy as np, pandas as pd
import pool_sources as SRC   # candidate-source generators
import photog_feats as PHO   # photographic per-crop features

mk_aspect = SRC.make_box_aspect
AREAS_MED = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]   # medium area grid
NMS_C = 0.95                                              # re-aspect NMS threshold

# ─── base geometry config (center-shift grid + small-area grid + rand) ───
C83_AREAS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.86, 0.90, 0.95]
C83_SHIFTS = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
C83_KEEP = 50
S2_AREAS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55]
S2_POS = [(0.0, 0.0), (0.10, 0.0), (-0.10, 0.0), (0.0, 0.10), (0.0, -0.10),
          (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15)]
S2_KEEP = 30
RAND_N = 40
NMS_BASE = 0.9
# LOCKED deterministic production seed. The frozen benchmark cache built `rand` with a batch-RNG
# advancing across the whole dataset (non-reproducible per-image). This release uses a fixed
# per-image seed so the pool is a deterministic function of one image -> byte-exact for any input.
RAND_SEED = 42


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter)


def _nms_rowwise(boxes_src, W, H, thr):
    """Greedy NMS in generation order: suppress a later box if IoU >= thr with any kept one."""
    keep = []
    for b, src in boxes_src:
        bn = (b[0]/W, b[1]/H, b[2]/W, b[3]/H)
        if all(_iou(bn, (k[0]/W, k[1]/H, k[2]/W, k[3]/H)) < thr for k, _ in keep):
            keep.append((b, src))
    return keep


def _box_area(W, H, area, cx, cy):
    r = math.sqrt(area); cw = int(W*r); ch = int(H*r)
    x1 = int(round(cx*W - cw/2)); y1 = int(round(cy*H - ch/2))
    x1 = max(0, min(x1, W-cw)); y1 = max(0, min(y1, H-ch))
    return (x1, y1, x1+cw, y1+ch)


def _nms_keep(raw, W, H, target, seed):
    """raw: list of dict(area,cx_frac,cy_frac,box). greedy NMS@NMS_BASE keep <=target."""
    if len(raw) <= target: return raw
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    order = rng.permutation(len(raw))
    kept, kept_b = [], []
    for i in order:
        b = raw[i]["box"]
        if all(_iou(b, kb) <= NMS_BASE for kb in kept_b):
            kept.append(raw[i]); kept_b.append(b)
        if len(kept) >= target: break
    return kept


def _base_geo(W, H):
    """c83_nms + small_area + rand base candidates (geometry only; feed C re-aspect)."""
    out = []
    # c83_nms
    raw, seen = [], set()
    for area in C83_AREAS:
        for sx in C83_SHIFTS:
            for sy in C83_SHIFTS:
                cx, cy = 0.5+sx, 0.5+sy
                if not (0 < cx < 1 and 0 < cy < 1): continue
                box = _box_area(W, H, area, cx, cy)
                if box in seen: continue
                seen.add(box); raw.append({"area": area, "cx_frac": cx, "cy_frac": cy, "box": box})
    out += _nms_keep(raw, W, H, C83_KEEP, hash((W, H)))
    # small_area
    raw = []
    for area in S2_AREAS:
        for sx, sy in S2_POS:
            cx, cy = 0.5+sx, 0.5+sy
            if not (0 < cx < 1 and 0 < cy < 1): continue
            raw.append({"area": area, "cx_frac": cx, "cy_frac": cy, "box": _box_area(W, H, area, cx, cy)})
    out += _nms_keep(raw, W, H, S2_KEEP, hash((W, H, 7)))
    # rand — LOCKED per-image deterministic seed (see RAND_SEED note above)
    rng = np.random.RandomState(RAND_SEED)
    n, tries = 0, 0
    while n < RAND_N and tries < RAND_N*10:
        tries += 1
        area = rng.uniform(0.15, 0.95); aspect = rng.uniform(0.5, 2.0)
        total = area*W*H; cw = max(20, min(int(math.sqrt(total*aspect)), W-1)); ch = max(20, min(int(total/max(cw, 1)), H-1))
        cx = rng.uniform(cw/2, W-cw/2); cy = rng.uniform(ch/2, H-ch/2)
        x1 = max(0, min(int(cx-cw/2), W-cw)); y1 = max(0, min(int(cy-ch/2), H-ch))
        out.append({"area": area, "cx_frac": (x1+cw/2)/W, "cy_frac": (y1+ch/2)/H, "box": (x1, y1, x1+cw, y1+ch)})
        n += 1
    return out


def _gen_C(base, W, H, ta):
    raw = []
    for c in base:
        area = max(0.15, min(float(c["area"]), 0.98))
        box = mk_aspect(W, H, area, ta, c["cx_frac"], c["cy_frac"])
        if box[2]-box[0] < 20 or box[3]-box[1] < 20: continue
        raw.append(box)
    if not raw: return []
    scores = np.array([(b[2]-b[0])*(b[3]-b[1]) for b in raw])
    order = np.argsort(-scores); kept, sup = [], set()
    for i in order:
        i = int(i)
        if i in sup: continue
        kept.append(raw[i])
        for j in order:
            j = int(j)
            if j == i or j in sup: continue
            if _iou(raw[i], raw[j]) > NMS_C: sup.add(j)
    return [(b, "C") for b in kept]


def _gen_inspy_anc(mask_np, W, H, ta):
    if mask_np is None or mask_np.sum() <= 1e-3: return []
    from scipy import ndimage
    m = mask_np.astype(np.float32)/255.0
    yy, xx = np.indices(m.shape); mt = float(m.sum())
    anchors = [("centroid", float((m*xx).sum()/mt)/max(m.shape[1]-1, 1), float((m*yy).sum()/mt)/max(m.shape[0]-1, 1))]
    bin_m = (m >= 0.5).astype(np.uint8)
    labeled, n_comp = ndimage.label(bin_m)
    blobs = []
    for k in range(1, n_comp+1):
        comp = (labeled == k); size = comp.sum()
        if size < 8: continue
        yb, xb = np.indices(comp.shape)
        blobs.append((float(size), (xb*comp).sum()/size/max(comp.shape[1]-1, 1), (yb*comp).sum()/size/max(comp.shape[0]-1, 1)))
    blobs.sort(key=lambda x: -x[0])
    for bi, (_, cxb, cyb) in enumerate(blobs[:3]):
        anchors.append((f"blob_{bi}", float(cxb), float(cyb)))
    if bin_m.sum() > 4:
        rows_m = np.any(bin_m, axis=1); cols_m = np.any(bin_m, axis=0)
        py1, py2 = np.where(rows_m)[0][[0, -1]]; px1, px2 = np.where(cols_m)[0][[0, -1]]
        anchors.append(("mbbox", float((px1+px2)/2/max(m.shape[1]-1, 1)), float((py1+py2)/2/max(m.shape[0]-1, 1))))
    out = []
    for _, cx, cy in anchors:
        for area in AREAS_MED:
            box = mk_aspect(W, H, area, ta, cx, cy)
            if box[2]-box[0] < 20 or box[3]-box[1] < 20: continue
            out.append((box, "INSPY_anc"))
    return out


def _gen_vmix_peak(vmix_2d, W, H, ta):
    if vmix_2d is None: return []
    h_p, w_p = vmix_2d.shape
    flat = vmix_2d.flatten(); topk = np.argsort(-flat)[:5]
    out = []
    for idx in topk:
        py, px = idx // w_p, idx % w_p
        cx = (px+0.5)/w_p; cy = (py+0.5)/h_p
        for area in AREAS_MED:
            box = mk_aspect(W, H, area, ta, cx, cy)
            if box[2]-box[0] < 20 or box[3]-box[1] < 20: continue
            out.append((box, "VMIX_peak"))
    return out


def _gen_multiscale(df_civ, W, H, ta):
    """Top-20 by vmix_cov from C+INSPY_anc+VMIX_peak × {0.7,0.85,1.15,1.3} area scales."""
    if len(df_civ) == 0: return []
    top20 = df_civ.nlargest(20, "vmix_cov")
    boxes = []
    for r in top20.itertuples(index=False):
        cx = (r.x1+r.x2)/2/W; cy = (r.y1+r.y2)/2/H
        base_area = (r.x2-r.x1)*(r.y2-r.y1)/max(W*H, 1)
        for scale in [0.7, 0.85, 1.15, 1.3]:
            area = float(np.clip(base_area*scale, 0.20, 0.95))
            box = mk_aspect(W, H, area, ta, cx, cy)
            if box[2]-box[0] < 20 or box[3]-box[1] < 20: continue
            boxes.append((box, "multiscale"))
    return boxes


def _feat_rows(boxes_src, pid, W, H, ta, vmix_2d, v1_2d, mask_np, mt, image_path):
    rows = []
    for box, src in boxes_src:
        f = SRC.compute_features_for_box(box, W, H, None, ta, vmix_2d, v1_2d, mask_np, mt)
        f.update({"photo_id": pid or 0, "image_path": str(image_path), "src": src, "img_w": W, "img_h": H})
        rows.append(f)
    return rows


def generate_pool(pid, W, H, target_aspect, vmix_2d, v1_2d, mask_np, rgb, image_path):
    """Build the 8-source production-aligned candidate pool for one image."""
    W = int(W); H = int(H); ta = float(target_aspect)
    mt = float(mask_np.sum()) + 1e-9 if mask_np is not None else 1e-9

    # C / INSPY_anc / VMIX_peak first (needed for multiscale top-20 by vmix_cov)
    civ = _gen_C(_base_geo(W, H), W, H, ta) + _gen_inspy_anc(mask_np, W, H, ta) + _gen_vmix_peak(vmix_2d, W, H, ta)
    df_civ = pd.DataFrame(_feat_rows(civ, pid, W, H, ta, vmix_2d, v1_2d, mask_np, mt, image_path))

    ms = _gen_multiscale(df_civ, W, H, ta)
    df_ms = pd.DataFrame(_feat_rows(ms, pid, W, H, ta, vmix_2d, v1_2d, mask_np, mt, image_path))

    # composition-principle S_* sources — per-source NMS@0.95 in generation order
    s_boxes = []
    for fn in [SRC.gen_source_diagonal, SRC.gen_source_negspace, SRC.gen_source_leadingline, SRC.gen_source_boundary]:
        try:
            raw = [(box, src) for box, src, _cx, _cy in fn(pid or 0, W, H, ta, None, mask_np, vmix_2d, v1_2d)]
        except Exception as exc:
            print(f"  WARN pool source {fn.__name__} failed and contributes no candidates: {exc}", flush=True)
            raw = []
        s_boxes += _nms_rowwise(raw, W, H, 0.95)
    df_s = pd.DataFrame(_feat_rows(s_boxes, pid, W, H, ta, vmix_2d, v1_2d, mask_np, mt, image_path))

    df = pd.concat([d for d in [df_civ, df_ms, df_s] if len(d)], ignore_index=True)
    df = _add_photog_and_derived(df, W, H, vmix_2d, mask_np, rgb)
    return df


def _add_photog_and_derived(df, W, H, vmix_2d, mask_np, rgb):
    """16 photographic features + 6 derived geometric features — matches the benchmark pool."""
    vpk = PHO.get_vmix_top2_peaks_norm(vmix_2d)
    keys = ["sym_h_diff", "sym_v_diff", "sym_h_inspy", "sym_diag_diff", "tri_top_apex", "tri_bot_apex",
            "tri_balance", "color_var_lum", "color_warm_ratio", "color_sat_mean", "visual_weight_x",
            "visual_weight_y", "cnt_balance", "headroom_top", "lead_room", "subject_size_ratio"]
    pf = {k: [] for k in keys}
    msf = mask_np.astype(np.float32) if mask_np is not None else None
    for r in df.itertuples(index=False):
        x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
        pf["sym_h_diff"].append(PHO.sym_h_diff(vmix_2d, x1, y1, x2, y2, W, H))
        pf["sym_v_diff"].append(PHO.sym_v_diff(vmix_2d, x1, y1, x2, y2, W, H))
        pf["sym_h_inspy"].append(PHO.sym_h_diff(msf, x1, y1, x2, y2, W, H) if msf is not None else 0.0)
        pf["sym_diag_diff"].append(PHO.sym_diag_diff(vmix_2d, x1, y1, x2, y2, W, H))
        cxc, cyc = (x1+x2)/2/W, (y1+y2)/2/H
        pf["tri_top_apex"].append(PHO.tri_dist_apex(cxc, cyc, (0.5, 0.333)))
        pf["tri_bot_apex"].append(PHO.tri_dist_apex(cxc, cyc, (0.5, 0.667)))
        pf["tri_balance"].append(PHO.tri_3pt_area((cxc, cyc), vpk[0], vpk[1]))
        lv, wr, sm = PHO.color_stats_box(rgb, x1, y1, x2, y2, W, H)
        pf["color_var_lum"].append(lv); pf["color_warm_ratio"].append(wr); pf["color_sat_mean"].append(sm)
        ar = (x2-x1)*(y2-y1)/max(W*H, 1)
        pf["visual_weight_x"].append((cxc-0.5)*ar); pf["visual_weight_y"].append((cyc-0.5)*ar)
        pf["cnt_balance"].append(PHO.cnt_balance(vmix_2d, x1, y1, x2, y2, W, H))
        pf["headroom_top"].append(PHO.headroom_top(mask_np, x1, y1, x2, y2, W, H))
        pf["lead_room"].append(PHO.lead_room(vmix_2d, mask_np, x1, y1, x2, y2, W, H))
        pf["subject_size_ratio"].append(PHO.subject_size_ratio(mask_np, x1, y1, x2, y2, W, H))
    for k, v in pf.items(): df[k] = v
    df["thirds_align"] = df.apply(lambda r: SRC.thirds_align(r.cx_frac, r.cy_frac), axis=1).astype(np.float32)
    df["golden_align"] = df.apply(lambda r: SRC.golden_align(r.cx_frac, r.cy_frac), axis=1).astype(np.float32)
    df["boundary_touch"] = ((df.x1 < 5) | (df.y1 < 5) | (df.x2 > df.img_w-5) | (df.y2 > df.img_h-5)).astype(np.float32)
    df["aspect_skew"] = np.log(np.clip(df.cand_aspect, 1e-6, None)) - np.log(np.clip(df.gt_aspect, 1e-6, None))
    df["subject_centered"] = df.ca_dist
    df["neg_space_ratio"] = df.area_ratio / np.clip(df.inspy_capture/255.0 + 1e-6, 1e-6, None)
    return df
