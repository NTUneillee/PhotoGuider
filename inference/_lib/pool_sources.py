"""Candidate-source generators for the conditioned production pool.

The four S_* candidate source generators, the aspect-honoring box constructor and the per-box
base-feature computation used by _lib/pool_gen.py. Function bodies are byte-identical extracts
of the validated benchmark pool machinery."""
import math
import numpy as np


AREAS_5 = [0.30, 0.45, 0.60, 0.75, 0.90]


AREAS_4 = [0.35, 0.55, 0.75, 0.90]


THIRDS = [(1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)]


GOLDEN = [(0.382, 0.382), (0.618, 0.382), (0.382, 0.618), (0.618, 0.618)]


def iou_xyxy_pair(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1); ub = (bx2 - bx1) * (by2 - by1)
    un = ua + ub - inter
    return inter / un if un > 0 else 0.0


def make_box_aspect(W, H, area, aspect, cx_frac, cy_frac):
    total = area * W * H
    cw = max(16, min(int(math.sqrt(total * aspect)), W))
    ch = max(16, min(int(total / max(cw, 1)), H))
    cx = cx_frac * W; cy = cy_frac * H
    x1 = int(round(cx - cw / 2)); y1 = int(round(cy - ch / 2))
    x1 = max(0, min(x1, W - cw)); y1 = max(0, min(y1, H - ch))
    return (x1, y1, x1 + cw, y1 + ch)


def coverage_box(sal_2d, x1, y1, x2, y2, W, H):
    if sal_2d is None: return 0.0
    h_p, w_p = sal_2d.shape
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1))
    py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    return float(sal_2d[py1:py2, px1:px2].sum() / (sal_2d.sum() + 1e-9))


def max_in_box(sal_2d, x1, y1, x2, y2, W, H):
    if sal_2d is None: return 0.0
    h_p, w_p = sal_2d.shape
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1))
    py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    inner = sal_2d[py1:py2, px1:px2]
    return float(inner.max()) if inner.size else 0.0


def get_inspy_centroid(mask_np):
    if mask_np is None or mask_np.sum() < 1e-3:
        return 0.5, 0.5
    m = mask_np.astype(np.float32) / 255.0
    yy, xx = np.indices(m.shape)
    mt = float(m.sum())
    cy = float((m * yy).sum() / mt) / max(m.shape[0] - 1, 1)
    cx = float((m * xx).sum() / mt) / max(m.shape[1] - 1, 1)
    return cx, cy


def get_inspy_bbox(mask_np, thr=64):
    if mask_np is None or mask_np.sum() < 1e-3:
        return None
    bw = (mask_np >= thr)
    if not bw.any():
        return None
    ys, xs = np.where(bw)
    h, w = mask_np.shape
    return (xs.min() / w, ys.min() / h, xs.max() / w, ys.max() / h)


def gen_source_diagonal(pid, W, H, gt_aspect, gt_box, mask_np, vmix_2d, v1_2d):
    """B: diagonal anchors — 5 along each of 2 diagonals × 7 areas."""
    rows = []
    diag_pts = []
    for t in [0.25, 0.4, 0.5, 0.6, 0.75]:
        diag_pts.append((t, t))           # main diagonal
        diag_pts.append((t, 1 - t))       # anti diagonal
    for cx_f, cy_f in diag_pts:
        for area in AREAS_5:
            box = make_box_aspect(W, H, area, gt_aspect, cx_f, cy_f)
            if box[2] - box[0] < 20 or box[3] - box[1] < 20: continue
            rows.append((box, "S_diagonal", cx_f, cy_f))
    return rows


def gen_source_leadingline(pid, W, H, gt_aspect, gt_box, mask_np, vmix_2d, v1_2d):
    """D: leading line — V_MIX max gradient direction → vanishing point estimate."""
    if vmix_2d is None: return []
    # Use V_MIX peak as endpoint, and weighted centroid as start
    h_p, w_p = vmix_2d.shape
    sm = np.maximum(vmix_2d, 0)
    if sm.sum() < 1e-6: return []
    yy, xx = np.indices(sm.shape)
    mt = float(sm.sum())
    cy_w = float((sm * yy).sum() / mt) / max(h_p - 1, 1)
    cx_w = float((sm * xx).sum() / mt) / max(w_p - 1, 1)
    # Peak
    pi = int(np.argmax(sm))
    py, px = pi // w_p, pi % w_p
    px_n = px / max(w_p - 1, 1); py_n = py / max(h_p - 1, 1)
    rows = []
    # 3 sample points along the line from centroid to peak, extrapolate
    for t in [0.5, 1.0, 1.4]:
        cx_f = float(np.clip(cx_w + t * (px_n - cx_w), 0.1, 0.9))
        cy_f = float(np.clip(cy_w + t * (py_n - cy_w), 0.1, 0.9))
        for area in AREAS_5:
            box = make_box_aspect(W, H, area, gt_aspect, cx_f, cy_f)
            if box[2] - box[0] < 20 or box[3] - box[1] < 20: continue
            rows.append((box, "S_leading", cx_f, cy_f))
    return rows


def gen_source_negspace(pid, W, H, gt_aspect, gt_box, mask_np, vmix_2d, v1_2d):
    """E: negative-space — InSPy bbox × 5 expand × 5 shift."""
    bb = get_inspy_bbox(mask_np)
    if bb is None: return []
    bx1, by1, bx2, by2 = bb
    bcx = (bx1 + bx2) / 2; bcy = (by1 + by2) / 2
    bw = bx2 - bx1; bh = by2 - by1
    rows = []
    shifts = [(0, 0), (0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]
    for shift_x, shift_y in shifts:
        cx_f = float(np.clip(bcx + shift_x, 0.1, 0.9))
        cy_f = float(np.clip(bcy + shift_y, 0.1, 0.9))
        for expand in [1.3, 1.5, 1.8, 2.2, 2.7]:
            area = float(np.clip(bw * bh * expand, 0.20, 0.95))
            box = make_box_aspect(W, H, area, gt_aspect, cx_f, cy_f)
            if box[2] - box[0] < 20 or box[3] - box[1] < 20: continue
            rows.append((box, "S_negspace", cx_f, cy_f))
    return rows


def gen_source_boundary(pid, W, H, gt_aspect, gt_box, mask_np, vmix_2d, v1_2d):
    """H: boundary-touch — InSPy bbox shifted to touch each edge × 4 areas."""
    bb = get_inspy_bbox(mask_np)
    if bb is None: return []
    bx1, by1, bx2, by2 = bb
    bcx = (bx1 + bx2) / 2; bcy = (by1 + by2) / 2
    rows = []
    # Touch left/right/top/bottom — anchor moves so cand bbox touches
    for edge in ["left", "right", "top", "bottom"]:
        for area in AREAS_4:
            # We want crop to *contain* most of subject but extend to edge
            if edge == "left":
                cx_f, cy_f = max(bcx - 0.15, 0.15), bcy
            elif edge == "right":
                cx_f, cy_f = min(bcx + 0.15, 0.85), bcy
            elif edge == "top":
                cx_f, cy_f = bcx, max(bcy - 0.15, 0.15)
            else:
                cx_f, cy_f = bcx, min(bcy + 0.15, 0.85)
            cx_f = float(np.clip(cx_f, 0.1, 0.9))
            cy_f = float(np.clip(cy_f, 0.1, 0.9))
            box = make_box_aspect(W, H, area, gt_aspect, cx_f, cy_f)
            if box[2] - box[0] < 20 or box[3] - box[1] < 20: continue
            rows.append((box, "S_boundary", cx_f, cy_f))
    return rows


def compute_features_for_box(box, W, H, gt_box, gt_aspect,
                              vmix_2d, v1_2d, mask_np, mt):
    x1, y1, x2, y2 = box
    vmix_cov = coverage_box(vmix_2d, x1, y1, x2, y2, W, H)
    vmix_max = max_in_box(vmix_2d, x1, y1, x2, y2, W, H)
    v1_cov = coverage_box(v1_2d, x1, y1, x2, y2, W, H)
    v1_max = max_in_box(v1_2d, x1, y1, x2, y2, W, H)
    if mask_np is not None:
        h_p, w_p = mask_np.shape
        px1_ = max(0, min(int(x1 / W * w_p), w_p - 1))
        py1_ = max(0, min(int(y1 / H * h_p), h_p - 1))
        px2_ = max(px1_ + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
        py2_ = max(py1_ + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
        inner = mask_np[py1_:py2_, px1_:px2_].astype(np.float32) / 255.0
        inspy_capture = float(inner.sum() / mt) * 255.0
        inspy_max = float(inner.max()) if inner.size else 0.0
    else:
        inspy_capture = 0.0; inspy_max = 0.0
    cx_c = (x1 + x2) / 2.0 / W; cy_c = (y1 + y2) / 2.0 / H
    if mask_np is not None and mask_np.sum() > 1e-3:
        cx_m, cy_m = get_inspy_centroid(mask_np)
    else:
        cx_m, cy_m = 0.5, 0.5
    ca_dist = math.hypot(cx_c - cx_m, cy_c - cy_m)
    area_ratio = (x2 - x1) * (y2 - y1) / max(1, W * H)
    cw_v = x2 - x1; ch_v = y2 - y1
    c_aspect = cw_v / max(1e-6, ch_v)
    log_diff_aspect = abs(math.log(max(c_aspect, 1e-6)) -
                            math.log(max(gt_aspect, 1e-6)))
    return {
        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        "iou_gt": iou_xyxy_pair(box, gt_box) if gt_box else 0.0,
        "vmix_cov": vmix_cov, "vmix_max": vmix_max,
        "v1_cov": v1_cov, "v1_max": v1_max,
        "inspy_capture": inspy_capture, "inspy_max": inspy_max,
        "ca_dist": ca_dist, "area_ratio": area_ratio,
        "log_aspect_diff": log_diff_aspect,
        "cand_aspect": c_aspect, "gt_aspect": gt_aspect,
        "cx_frac": cx_c, "cy_frac": cy_c,
        "mask_total_norm": mt / (256 * 256) if mask_np is not None else 0.0,
    }


def thirds_align(cx_f, cy_f):
    return max(0.0, 1.0 - min((cx_f - tx) ** 2 + (cy_f - ty) ** 2
                                for tx, ty in THIRDS) ** 0.5 / 0.2)


def golden_align(cx_f, cy_f):
    return max(0.0, 1.0 - min((cx_f - tx) ** 2 + (cy_f - ty) ** 2
                                for tx, ty in GOLDEN) ** 0.5 / 0.2)
