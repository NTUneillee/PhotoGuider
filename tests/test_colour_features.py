#!/usr/bin/env python3
"""The per-candidate colour features are now reduced from whole-image per-pixel maps. They must be
bit-identical to computing each crop from scratch (the reference below is the previous code)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference' / '_lib'))
import photog_feats as PHO


def ref_stage_c(rgb, x1, y1, x2, y2):
    cy_arr = rgb[y1:y2, x1:x2]
    cy_f = cy_arr.astype(np.float32) / 255.0
    lum = 0.299*cy_f[:,:,0] + 0.587*cy_f[:,:,1] + 0.114*cy_f[:,:,2]
    mx = cy_f.max(axis=2); mn = cy_f.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    warm = (cy_arr[:,:,0].astype(int) + cy_arr[:,:,1].astype(int) > 2*cy_arr[:,:,2].astype(int)).mean()
    return float(lum.std()), float(sat.mean()), float(warm)


def ref_color_stats_box(rgb, x1, y1, x2, y2, W, H):
    h_p, w_p = rgb.shape[:2]
    px1 = max(0, min(int(x1 / W * w_p), w_p - 1)); py1 = max(0, min(int(y1 / H * h_p), h_p - 1))
    px2 = max(px1 + 1, min(int(np.ceil(x2 / W * w_p)), w_p)); py2 = max(py1 + 1, min(int(np.ceil(y2 / H * h_p)), h_p))
    sub = rgb[py1:py2, px1:px2]
    if sub.size < 9:
        return (0.0, 0.0, 0.0)
    r = sub[..., 0].astype(np.float32) / 255.0; g = sub[..., 1].astype(np.float32) / 255.0; b = sub[..., 2].astype(np.float32) / 255.0
    cmax = np.maximum(np.maximum(r, g), b); cmin = np.minimum(np.minimum(r, g), b); delta = cmax - cmin
    V = cmax; S = np.where(cmax > 0, delta / (cmax + 1e-9), 0.0)
    H_arr = np.zeros_like(V); nz = delta > 1e-6
    rmax = (cmax == r) & nz; gmax = (cmax == g) & nz; bmax = (cmax == b) & nz
    H_arr = np.where(rmax, ((g - b) / np.maximum(delta, 1e-9)) % 6, H_arr)
    H_arr = np.where(gmax, (b - r) / np.maximum(delta, 1e-9) + 2, H_arr)
    H_arr = np.where(bmax, (r - g) / np.maximum(delta, 1e-9) + 4, H_arr)
    H_arr = H_arr * 60.0; H_arr = np.where(H_arr < 0, H_arr + 360, H_arr)
    return (float(V.std()), float(((H_arr < 60) | (H_arr > 300)).sum() / V.size), float(S.mean()))


def boxes(rng, W, H, n):
    out = [(0, 0, W, H), (0, 0, 1, 1), (W - 3, H - 2, W, H), (5, 5, 6, 40)]
    for _ in range(n):
        x1, x2 = sorted(rng.integers(0, W, 2)); y1, y2 = sorted(rng.integers(0, H, 2))
        out.append((int(x1), int(y1), int(max(x2, x1 + 1)), int(max(y2, y1 + 1))))
    return out


def main():
    rng = np.random.default_rng(0)
    for (W, H) in ((1280, 960), (640, 427), (37, 23)):
        rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        rgb[:H // 3] = (200, 180, 40)                      # a flat warm band: saturation/hue edge cases
        lum_map, sat_map, warm_map = PHO.colour_maps(rgb)
        for (x1, y1, x2, y2) in boxes(rng, W, H, 300):
            lum = np.ascontiguousarray(lum_map[y1:y2, x1:x2])
            got = (float(lum.std()), float(np.ascontiguousarray(sat_map[y1:y2, x1:x2]).mean()),
                   float(np.ascontiguousarray(warm_map[y1:y2, x1:x2]).mean()))
            assert got == ref_stage_c(rgb, x1, y1, x2, y2), ("stage_c", W, H, x1, y1, x2, y2, got)
            Wf, Hf = float(W) * 1.5, float(H) * 1.5                 # pool boxes are in a scaled frame
            got = PHO.color_stats_box(rgb, x1 * 1.5, y1 * 1.5, x2 * 1.5, y2 * 1.5, Wf, Hf)
            assert got == ref_color_stats_box(rgb, x1 * 1.5, y1 * 1.5, x2 * 1.5, y2 * 1.5, Wf, Hf), ("pool", W, H, x1, y1, x2, y2, got)
        # the one-entry map cache must key on the array, not on equal content
        other = rgb.copy(); other[H // 2:] = 0
        assert PHO.color_stats_box(other, 0, 0, W, H, W, H) == ref_color_stats_box(other, 0, 0, W, H, W, H)
    assert PHO.color_stats_box(None, 0, 0, 1, 1, 1, 1) == (0.0, 0.0, 0.0)
    print("colour features: bit-identical on 3 image sizes x 304 boxes")


if __name__ == '__main__':
    main()
