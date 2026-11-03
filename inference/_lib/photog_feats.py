"""Photographic-composition per-crop features.

The 16 photographic feature functions (symmetry / triangle / color / balance / headroom /
lead-room / subject-size) used by _lib/pool_gen.py. Byte-identical extracts of the validated
benchmark feature code."""
import math
import numpy as np
from PIL import Image


def crop_pixel(arr2d, x1, y1, x2, y2, W, H):
    """Crop sub-array of arr2d (h_p × w_p × ?) by px-coord on (W, H)."""
    if arr2d is None: return None
    h_p = arr2d.shape[0]; w_p = arr2d.shape[1]
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1))
    py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    return arr2d[py1:py2, px1:px2]


def sym_h_diff(sal2d, x1, y1, x2, y2, W, H):
    s = crop_pixel(sal2d, x1, y1, x2, y2, W, H)
    if s is None or s.size < 4: return 0.0
    mid = s.shape[1] // 2
    if mid == 0: return 0.0
    left = s[:, :mid].sum(); right = s[:, mid:2*mid].sum()
    total = left + right
    return float(abs(left - right) / max(total, 1e-9))


def sym_v_diff(sal2d, x1, y1, x2, y2, W, H):
    s = crop_pixel(sal2d, x1, y1, x2, y2, W, H)
    if s is None or s.size < 4: return 0.0
    mid = s.shape[0] // 2
    if mid == 0: return 0.0
    top = s[:mid, :].sum(); bot = s[mid:2*mid, :].sum()
    total = top + bot
    return float(abs(top - bot) / max(total, 1e-9))


def sym_diag_diff(sal2d, x1, y1, x2, y2, W, H):
    s = crop_pixel(sal2d, x1, y1, x2, y2, W, H)
    if s is None or s.size < 4: return 0.0
    h, w = s.shape
    n = min(h, w)
    if n < 2: return 0.0
    # Compare TL diagonal half with BR diagonal half via flip
    s_sq = s[:n, :n]
    s_flip = s_sq.T  # transpose = flip over main diagonal
    diff = float(np.abs(s_sq - s_flip).sum() / (s_sq.sum() + s_flip.sum() + 1e-9))
    return diff


def tri_dist_apex(cx_m, cy_m, apex):
    return float(math.hypot(cx_m - apex[0], cy_m - apex[1]))


def tri_3pt_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                      (p3[0] - p1[0]) * (p2[1] - p1[1]))


def get_vmix_top2_peaks_norm(sal2d):
    if sal2d is None: return [(0.5, 0.5), (0.5, 0.5)]
    sm = sal2d.copy()
    h_p, w_p = sm.shape
    peaks = []
    for _ in range(2):
        i = int(np.argmax(sm))
        py, px = i // w_p, i % w_p
        peaks.append((px / max(w_p - 1, 1), py / max(h_p - 1, 1)))
        py1 = max(0, py - 2); py2 = min(h_p, py + 3)
        px1 = max(0, px - 2); px2 = min(w_p, px + 3)
        sm[py1:py2, px1:px2] = -np.inf
    return peaks


def color_stats_box(rgb, x1, y1, x2, y2, W, H):
    """rgb: H×W×3 uint8 PIL array. Returns (lum_var, warm_ratio, sat_mean)."""
    if rgb is None:
        return (0.0, 0.0, 0.0)
    h_p, w_p = rgb.shape[:2]
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1))
    py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    sub = rgb[py1:py2, px1:px2]
    if sub.size < 9:
        return (0.0, 0.0, 0.0)
    # Convert to HSV (cheap via numpy)
    r = sub[..., 0].astype(np.float32) / 255.0
    g = sub[..., 1].astype(np.float32) / 255.0
    b = sub[..., 2].astype(np.float32) / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    V = cmax
    S = np.where(cmax > 0, delta / (cmax + 1e-9), 0.0)
    # Hue (0-360)
    H_arr = np.zeros_like(V)
    nz = delta > 1e-6
    rmax = (cmax == r) & nz
    gmax = (cmax == g) & nz
    bmax = (cmax == b) & nz
    H_arr = np.where(rmax, ((g - b) / np.maximum(delta, 1e-9)) % 6, H_arr)
    H_arr = np.where(gmax, (b - r) / np.maximum(delta, 1e-9) + 2, H_arr)
    H_arr = np.where(bmax, (r - g) / np.maximum(delta, 1e-9) + 4, H_arr)
    H_arr = H_arr * 60.0
    H_arr = np.where(H_arr < 0, H_arr + 360, H_arr)
    lum_var = float(V.std())
    warm_ratio = float(((H_arr < 60) | (H_arr > 300)).sum() / V.size)
    sat_mean = float(S.mean())
    return (lum_var, warm_ratio, sat_mean)


def cnt_balance(sal2d, x1, y1, x2, y2, W, H):
    s = crop_pixel(sal2d, x1, y1, x2, y2, W, H)
    if s is None or s.size < 4 or s.sum() < 1e-9: return 0.0
    h, w = s.shape
    yy, xx = np.indices(s.shape)
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0
    dist = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    weighted = float((s * dist).sum() / s.sum())
    return weighted / max(math.hypot(xc, yc), 1.0)


def headroom_top(mask_np, x1, y1, x2, y2, W, H):
    if mask_np is None or mask_np.sum() < 1e-3: return 0.0
    h_p, w_p = mask_np.shape
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1))
    py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    sub = mask_np[py1:py2, px1:px2]
    if sub.size < 4: return 0.0
    bw = sub >= 64
    if not bw.any(): return 0.0
    rows_with_mask = np.where(bw.any(axis=1))[0]
    if len(rows_with_mask) == 0: return 0.0
    top_mask = rows_with_mask[0]
    return float(top_mask / max(sub.shape[0] - 1, 1))


def lead_room(sal2d, mask_np, x1, y1, x2, y2, W, H):
    """Approximate lead room: side with more saliency outside mask."""
    if sal2d is None or mask_np is None or mask_np.sum() < 1e-3: return 0.0
    s = crop_pixel(sal2d, x1, y1, x2, y2, W, H)
    m = crop_pixel(mask_np, x1, y1, x2, y2, W, H)
    if s is None or m is None or s.size < 4: return 0.0
    if m.shape != s.shape:
        m_pil = Image.fromarray(m).resize((s.shape[1], s.shape[0]), Image.BILINEAR)
        m = np.asarray(m_pil)
    bw = m >= 64
    if not bw.any(): return 0.0
    cols_with_mask = np.where(bw.any(axis=0))[0]
    if len(cols_with_mask) == 0: return 0.0
    cl = cols_with_mask[0]
    cr = cols_with_mask[-1]
    left_sum = float(s[:, :cl].sum()) if cl > 0 else 0.0
    right_sum = float(s[:, cr+1:].sum()) if cr < s.shape[1] - 1 else 0.0
    total = left_sum + right_sum + 1e-9
    return float(max(left_sum, right_sum) / total)


def subject_size_ratio(mask_np, x1, y1, x2, y2, W, H):
    if mask_np is None or mask_np.sum() < 1e-3: return 0.0
    h_p, w_p = mask_np.shape
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1))
    py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p))
    py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    sub = mask_np[py1:py2, px1:px2]
    if sub.size < 4: return 0.0
    mask_pix = (sub >= 64).sum()
    return float(mask_pix / max(sub.size, 1))
