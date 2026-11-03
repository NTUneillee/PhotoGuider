"""Panorama geometry and the coarse direction search, as pure functions.

    view = render_view(pano_bgr, pan, tilt, fov, w, h)          # equirect -> pinhole view
    hotspots = select_hotspots(scores, pans, tilts)              # scored grid -> [(pan, tilt, score), ...]
    cam = crop_to_camera(box, pan0, tilt0)                       # working-view box -> (pan, tilt, fov, aspect)
    w, h = final_size(cam["aspect"])                             # native render size
    view, gain = render_hdr_view(load_hdr(path), pan, tilt, fov, w, h)   # per-view exposure from the HDR

Conventions (OUTPUT_SCHEMA.md): camera x right, y up, z forward, rotation Ry(pan) @ Rx(-tilt);
pan 0 faces the panorama centre column and increases toward increasing columns; tilt is + up.
The working view is a 60 degree pinhole of 1600 x 1067; the final render keeps the crop's ratio
at a 1920-pixel long side. Values here are the ones the paper reports (coarse grid 24 x 5 at
fov 40, 1024 x 1024 views).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# coarse scan grid
SCAN_PANS = list(range(0, 360, 15))        # 24 pans
SCAN_TILTS = list(range(-30, 31, 15))      # 5 tilts
SCAN_FOV, SCAN_SIZE = 40.0, 1024
# working view and final render
FOV0 = 60.0
W0, H0 = 1600, 1067
FINAL_LONG = 1920
# hotspot selection
NMS_RADIUS_DEG = 20.0        # angular suppression radius (~fov/2)
ADAPTIVE_FRAC = 0.15         # hotspot band = fraction of the field range below the maximum
K_MAX = 3
DPP_SIGMA_DEG = 25.0


# ───────────────────────────── rotations and reprojection ─────────────────────────────
def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def view_maps(pano_w, pano_h, pan_deg, tilt_deg, fov_deg, out_w, out_h):
    """(map_x, map_y) float32 remap grids sampling the equirect for one perspective view."""
    fov = np.deg2rad(fov_deg)
    focal = (out_w / 2.0) / np.tan(fov / 2.0)
    us = np.arange(out_w, dtype=np.float64) - (out_w - 1) / 2.0
    vs = np.arange(out_h, dtype=np.float64) - (out_h - 1) / 2.0
    uu, vv = np.meshgrid(us, vs)                       # image u-right, v-down
    dirs = np.stack([uu, -vv, np.full_like(uu, focal)], axis=-1)  # camera x right, y up, z forward
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    R = _rot_y(np.deg2rad(pan_deg)) @ _rot_x(np.deg2rad(-tilt_deg))
    world = dirs @ R.T
    dx, dy, dz = world[..., 0], world[..., 1], world[..., 2]
    lon = np.arctan2(dx, dz)                            # [-pi, pi], 0 = forward
    lat = np.arcsin(np.clip(dy, -1, 1))                # [-pi/2, pi/2], + = up
    map_x = (lon / (2 * np.pi) + 0.5) * pano_w
    map_y = (0.5 - lat / np.pi) * pano_h
    return map_x.astype(np.float32), map_y.astype(np.float32)


def render_view(pano_bgr, pan_deg, tilt_deg, fov_deg=SCAN_FOV, out_w=SCAN_SIZE, out_h=SCAN_SIZE):
    """Render one perspective view from an equirect BGR image (uint8); the seam wraps."""
    H, W = pano_bgr.shape[:2]
    mx, my = view_maps(W, H, pan_deg, tilt_deg, fov_deg, out_w, out_h)
    return cv2.remap(pano_bgr, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ───────────────────────────── per-view exposure from the linear HDR ─────────────────────────────
# The coarse scan scores the globally tonemapped 8-bit panorama (one exposure for every view, so view
# scores are comparable). Given the source HDR, the working view and the final shot are rendered
# from it and exposed for that view: median luminance -> middle grey, then the AgX curve of the
# global tonemap. The final shot reuses the working view's gain, so the crop decision and the
# delivered tones agree.
TARGET_GREY = 0.18


def load_hdr(path):
    """Linear float32 BGR equirect from a Radiance .hdr (non-finite and negative values -> 0)."""
    hdr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if hdr is None:
        raise SystemExit(f"cannot read HDR panorama: {path}")
    return np.maximum(np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), 0)


def meter_gain(view_hdr, target=TARGET_GREY):
    """Exposure gain that puts the view's median luminance at middle grey; returns (gain, median)."""
    lum = 0.2126 * view_hdr[..., 2] + 0.7152 * view_hdr[..., 1] + 0.0722 * view_hdr[..., 0]
    pos = lum[lum > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    return target / max(med, 1e-8), med


def agx_curve(x):
    """AgX tone curve on exposure-adjusted linear BGR -> display BGR in [0, 1]."""
    rgb = x[..., ::-1].copy()
    m_in = np.array([[0.842479062253094, 0.0784335999999992, 0.0792237451477643],
                     [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
                     [0.0423756549057051, 0.0784336, 0.879142973793104]])
    m_out = np.array([[1.19687900512017, -0.0980208811401368, -0.0990297440797205],
                      [-0.0528968517574562, 1.15190312990417, -0.0989611768448433],
                      [-0.0529716355144438, -0.0980434501171241, 1.15107367264116]])
    v = np.clip(rgb @ m_in.T, 1e-10, None)
    lv = np.clip((np.log2(v) + 12.47393) / (4.026069 + 12.47393), 0, 1)      # log2 over ~16.5 stops
    z = (((((15.5 * lv - 40.14) * lv + 31.96) * lv - 6.868) * lv + 0.4298) * lv + 0.1191) * lv - 0.00232
    return np.clip(np.clip(z, 0, 1) @ m_out.T, 0, 1)[..., ::-1]


def render_hdr_view(hdr, pan_deg, tilt_deg, fov_deg, out_w, out_h, gain=None):
    """Perspective view from the linear HDR, exposed with ``gain`` (metered on the view if None).
    Returns (8-bit BGR view, gain)."""
    mx, my = view_maps(hdr.shape[1], hdr.shape[0], pan_deg, tilt_deg, fov_deg, out_w, out_h)
    view = cv2.remap(hdr, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if gain is None:
        gain, _ = meter_gain(view)
    return (np.clip(agx_curve(view * gain), 0, 1) * 255).astype(np.uint8), gain


# ───────────────────────────── crop box -> camera ─────────────────────────────
def _ray_cam(u, v, focal, cx, cy):
    d = np.array([u - cx, -(v - cy), focal], dtype=np.float64)
    return d / np.linalg.norm(d)


def _dir_to_pantilt(d):
    pan = np.degrees(np.arctan2(d[0], d[2]))
    tilt = np.degrees(np.arcsin(np.clip(d[1], -1, 1)))
    return float(pan), float(tilt)


def crop_to_camera(box, pan0, tilt0):
    """Camera parameters that reproduce a crop ``box`` of the working view at (pan0, tilt0):
    the crop centre ray gives (pan, tilt), the two horizontal edge rays give the field of view,
    and the crop's pixel ratio is the output aspect. The re-render is an approximation of the
    crop (corner rays differ by a few degrees at large tilts)."""
    x0, y0, x1, y1 = box
    focal = (W0 / 2.0) / np.tan(np.deg2rad(FOV0) / 2.0)
    cx, cy = (W0 - 1) / 2.0, (H0 - 1) / 2.0
    R = _rot_y(np.deg2rad(pan0)) @ _rot_x(np.deg2rad(-tilt0))
    uc, vc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pan_p, tilt_p = _dir_to_pantilt(R @ _ray_cam(uc, vc, focal, cx, cy))
    a = _ray_cam(x0, vc, focal, cx, cy)
    b = _ray_cam(x1, vc, focal, cx, cy)
    fov_p = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    return dict(pan=pan_p, tilt=tilt_p, fov=fov_p, aspect=float((x1 - x0) / (y1 - y0)))


def final_size(aspect):
    if aspect >= 1.0:
        return FINAL_LONG, int(round(FINAL_LONG / aspect))
    return int(round(FINAL_LONG * aspect)), FINAL_LONG


# ───────────────────────────── direction search ─────────────────────────────
def _unit(pan_deg, tilt_deg):
    lon, lat = np.deg2rad(pan_deg), np.deg2rad(tilt_deg)
    return np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])


def gc_angle(p1, t1, p2, t2):
    """Great-circle angle in degrees between two look directions."""
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(p1, t1), _unit(p2, t2)), -1, 1))))


def score_field(scores, pans, tilts):
    """Interpolate the scored (pan, tilt) grid onto a dense 1-degree field over the scanned band.
    ``scores``: DataFrame with pan, tilt, comp_score. Returns (pan_dense, tilt_dense, field[t, p], fmin, fmax)."""
    pans = np.array(pans, float); tilts = np.array(tilts, float)
    grid = np.full((len(tilts), len(pans)), np.nan)
    for _, r in scores.iterrows():
        grid[tilts.tolist().index(r["tilt"]), pans.tolist().index(r["pan"])] = r["comp_score"]
    if np.isnan(grid).any():
        raise ValueError("score grid has unscored (pan, tilt) cells")
    pans_ext = np.concatenate([pans, [360.0]])          # pan wraps
    grid_ext = np.concatenate([grid, grid[:, :1]], axis=1)
    interp = RegularGridInterpolator((tilts, pans_ext), grid_ext, bounds_error=False, fill_value=None)
    pan_dense = np.arange(0.0, 360.0, 1.0)
    tilt_dense = np.arange(tilts.min(), tilts.max() + 1e-6, 1.0)
    lon, lat = np.meshgrid(pan_dense, tilt_dense)
    field = interp(np.stack([lat.ravel(), lon.ravel()], -1)).reshape(lat.shape)
    return pan_dense, tilt_dense, field, float(field.min()), float(field.max())


def local_maxima(pan_dense, tilt_dense, field):
    """Cells >= all valid 8-neighbours (pan wraps, tilt clamps) -> [(pan, tilt, score), ...]."""
    nt, npn = field.shape
    peaks = []
    for i in range(nt):
        for j in range(npn):
            v = field[i, j]
            is_peak = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii = i + di
                    if ii < 0 or ii >= nt:
                        continue
                    if field[ii, (j + dj) % npn] > v:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if is_peak:
                peaks.append((float(pan_dense[j]), float(tilt_dense[i]), float(v)))
    return peaks


def angular_nms(peaks, radius=NMS_RADIUS_DEG):
    """Keep the highest-scoring peaks first; drop any within ``radius`` degrees of a kept one."""
    kept = []
    for pan, tilt, score in sorted(peaks, key=lambda x: -x[2]):
        if all(gc_angle(pan, tilt, kp, kt) >= radius for kp, kt, _ in kept):
            kept.append((pan, tilt, score))
    return kept


def dpp_pick(cands, k, fmin, fmax, sigma=DPP_SIGMA_DEG):
    """Greedy DPP MAP over candidate directions: quality = normalized score, similarity =
    angular Gaussian of width ``sigma``."""
    q = np.array([(c[2] - fmin) / (fmax - fmin + 1e-9) for c in cands])
    n = len(cands)
    S = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            ang = gc_angle(cands[a][0], cands[a][1], cands[b][0], cands[b][1])
            S[a, b] = S[b, a] = np.exp(-(ang ** 2) / (sigma ** 2))
    L = np.outer(q, q) * S
    selected, remaining = [], list(range(n))
    while len(selected) < k and remaining:
        best_j, best_det = None, -np.inf
        for j in remaining:
            det = np.linalg.det(L[np.ix_(selected + [j], selected + [j])])
            if det > best_det:
                best_det, best_j = det, j
        if best_j is None or best_det <= 0:
            break
        selected.append(best_j)
        remaining.remove(best_j)
    return [cands[j] for j in selected]


def select_hotspots(scores, pans=SCAN_PANS, tilts=SCAN_TILTS):
    """Ordered shooting directions [(pan, tilt, score), ...] (rank 0 = strongest) from the scored
    grid: dense field -> local maxima -> angular NMS -> adaptive band -> DPP to at most K_MAX."""
    pan_d, tilt_d, field, fmin, fmax = score_field(scores, pans, tilts)
    if fmax - fmin <= 1e-12:
        raise ValueError(f"view scores are flat ({fmax:.4g} everywhere): the scorer expressed no preference")
    peaks = angular_nms(local_maxima(pan_d, tilt_d, field))
    thr = fmax - ADAPTIVE_FRAC * (fmax - fmin)
    cands = [p for p in peaks if p[2] >= thr]
    hotspots = dpp_pick(cands, K_MAX, fmin, fmax) if len(cands) > K_MAX else cands
    return sorted(hotspots, key=lambda x: -x[2])
