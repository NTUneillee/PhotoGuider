"""Twin-candidate per-crop featurizer.

The verbatim per-crop feature loop the twin-candidate manifests are featurized with (the
canonical benchmark path); the saliency / attention / mask maps are passed in memory. Byte-identical extract."""
import math
import time
import numpy as np
import pandas as pd
from PIL import Image


from pool_sources import coverage_box, max_in_box, get_inspy_centroid, thirds_align, golden_align
from photog_feats import crop_pixel


def stage_c_features(df_pool, maps):
    """Core-feature family for twin / dashboard / panorama rows. ``maps`` = (vmix, v1, mask) dicts
    photo_id -> (_, _, 2-D array), one entry per photo_id in ``df_pool``. Returns a DataFrame of the
    features plus the row metadata columns."""
    sal_dv3, sal_v1, inspy = maps

    n = len(df_pool)
    # Pre-allocate feature arrays
    fcols = ["vmix_max", "vmix_cov", "v1_max", "v1_cov", "inspy_capture", "inspy_max",
              "ca_dist", "area_ratio", "log_aspect_diff",
              "cand_aspect", "cx_frac", "cy_frac", "mask_total_norm",
              "thirds_align", "golden_align", "boundary_touch", "aspect_skew",
              "subject_centered", "neg_space_ratio",
              "sym_h_diff", "sym_v_diff", "sym_h_inspy", "sym_diag_diff",
              "tri_top_apex", "tri_bot_apex", "tri_balance",
              "color_var_lum", "color_warm_ratio", "color_sat_mean",
              "visual_weight_x", "visual_weight_y", "cnt_balance",
              "headroom_top", "lead_room", "subject_size_ratio"]
    feats = {k: np.zeros(n, dtype=np.float32) for k in fcols}

    # Per-pid caches (saliency + rgb + centroid)
    rgb_cache = {}
    last_pid = None
    vmix_2d = None; v1_2d = None; mask_np = None; rgb = None
    cx_m, cy_m = 0.5, 0.5
    mt = 1.0
    vmix_peaks = [(0.5, 0.5), (0.5, 0.5)]

    t0 = time.time()
    for i, r in enumerate(df_pool.itertuples(index=False)):
        pid = r.photo_id
        if pid != last_pid:
            dv3_pkg = sal_dv3.get(pid)        # V_MIX_sum
            v1_pkg = sal_v1.get(pid)          # V1 (separate)
            inspy_pkg = inspy.get(pid)
            vmix_2d = dv3_pkg[2] if dv3_pkg else None
            v1_2d = v1_pkg[2] if v1_pkg else None
            mask_np = inspy_pkg[2] if inspy_pkg else None
            cx_m, cy_m = get_inspy_centroid(mask_np)
            mt = float(mask_np.astype(np.float32).sum() / 255.0) if mask_np is not None else 1.0
            if mt < 1e-3: mt = 1.0
            # V_MIX top-2 peaks for tri_balance
            if vmix_2d is not None:
                sm = vmix_2d.copy(); h_p, w_p = sm.shape
                peaks = []
                for _ in range(2):
                    pi_idx = int(np.argmax(sm))
                    py, px = pi_idx // w_p, pi_idx % w_p
                    peaks.append((px / max(w_p-1, 1), py / max(h_p-1, 1)))
                    py1=max(0, py-2); py2=min(h_p, py+3); px1=max(0, px-2); px2=min(w_p, px+3)
                    sm[py1:py2, px1:px2] = -np.inf
                vmix_peaks = peaks
            # RGB image (cached up to 50 pids in flight)
            if pid not in rgb_cache:
                try:
                    rgb_cache[pid] = np.asarray(Image.open(r.image_path).convert("RGB"))
                except Exception:
                    rgb_cache[pid] = None
                if len(rgb_cache) > 100:
                    rgb_cache = {k: v for k, v in list(rgb_cache.items())[-50:]}
            rgb = rgb_cache.get(pid)
            last_pid = pid

        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        W, H = float(r.img_w), float(r.img_h)
        gt_aspect = float(r.gt_aspect)

        # F_BASE 9
        feats["vmix_max"][i] = max_in_box(vmix_2d, x1, y1, x2, y2, W, H)
        feats["vmix_cov"][i] = coverage_box(vmix_2d, x1, y1, x2, y2, W, H)
        feats["v1_max"][i] = max_in_box(v1_2d, x1, y1, x2, y2, W, H)
        feats["v1_cov"][i] = coverage_box(v1_2d, x1, y1, x2, y2, W, H)
        if mask_np is not None:
            inner = crop_pixel(mask_np, x1, y1, x2, y2, W, H)
            if inner is not None and inner.size > 0:
                inner_f = inner.astype(np.float32) / 255.0
                feats["inspy_capture"][i] = float(inner_f.sum() / mt) * 255.0
                feats["inspy_max"][i] = float(inner.max())
        cx_c = (x1 + x2) / 2 / W; cy_c = (y1 + y2) / 2 / H
        feats["ca_dist"][i] = math.hypot(cx_c - cx_m, cy_c - cy_m)
        area_ratio = (x2-x1) * (y2-y1) / max(1, W*H)
        feats["area_ratio"][i] = area_ratio
        cw_v = x2 - x1; ch_v = y2 - y1
        c_aspect = cw_v / max(1e-6, ch_v)
        feats["log_aspect_diff"][i] = abs(math.log(max(c_aspect, 1e-6)) -
                                           math.log(max(gt_aspect, 1e-6)))
        feats["cand_aspect"][i] = c_aspect
        feats["cx_frac"][i] = cx_c; feats["cy_frac"][i] = cy_c
        feats["mask_total_norm"][i] = mt / (256*256) if mask_np is not None else 0.0

        # F_EXT_V1 add 6
        feats["thirds_align"][i] = thirds_align(cx_c, cy_c)
        feats["golden_align"][i] = golden_align(cx_c, cy_c)
        feats["boundary_touch"][i] = float(x1 < 5 or y1 < 5 or x2 > W-5 or y2 > H-5)
        feats["aspect_skew"][i] = abs(c_aspect - W/H) / max(W/H, 1e-6)
        feats["subject_centered"][i] = float(math.hypot(cx_c - cx_m, cy_c - cy_m) < 0.2)
        # neg_space_ratio: 1 - inspy_capture
        feats["neg_space_ratio"][i] = 1.0 - (feats["inspy_capture"][i] / 255.0)

        # F_NEW_PHOTOG 16
        # Symmetry (vmix_2d-based)
        crop_v = crop_pixel(vmix_2d, x1, y1, x2, y2, W, H)
        if crop_v is not None and crop_v.size > 4:
            flip_h = crop_v[:, ::-1]
            feats["sym_h_diff"][i] = float(np.abs(crop_v - flip_h).mean() / (crop_v.mean() + 1e-6))
            flip_v = crop_v[::-1]
            feats["sym_v_diff"][i] = float(np.abs(crop_v - flip_v).mean() / (crop_v.mean() + 1e-6))
            if crop_v.shape[0] == crop_v.shape[1]:
                flip_d = crop_v.T
                feats["sym_diag_diff"][i] = float(np.abs(crop_v - flip_d).mean() / (crop_v.mean() + 1e-6))
        if mask_np is not None:
            inner = crop_pixel(mask_np, x1, y1, x2, y2, W, H)
            if inner is not None and inner.size > 4:
                inner_f = inner.astype(np.float32)
                flip_h = inner_f[:, ::-1]
                feats["sym_h_inspy"][i] = float(np.abs(inner_f - flip_h).mean() / (inner_f.mean() + 1e-6))

        # Triangular
        feats["tri_top_apex"][i] = math.hypot(cx_c - 0.5, cy_c - 0.333)
        feats["tri_bot_apex"][i] = math.hypot(cx_c - 0.5, cy_c - 0.667)
        # tri_balance: area of triangle (cand_center, peak1, peak2)
        p1, p2 = vmix_peaks
        tb = abs((cx_c*(p1[1]-p2[1]) + p1[0]*(p2[1]-cy_c) + p2[0]*(cy_c-p1[1])) / 2.0)
        feats["tri_balance"][i] = tb

        # Color (RGB-based)
        if rgb is not None:
            cy_arr = rgb[y1:y2, x1:x2]
            if cy_arr.size > 0:
                cy_f = cy_arr.astype(np.float32) / 255.0
                lum = 0.299*cy_f[:,:,0] + 0.587*cy_f[:,:,1] + 0.114*cy_f[:,:,2]
                feats["color_var_lum"][i] = float(lum.std())
                mx = cy_f.max(axis=2); mn = cy_f.min(axis=2)
                sat = (mx - mn) / (mx + 1e-6)
                feats["color_sat_mean"][i] = float(sat.mean())
                # warm ratio: pixels where R+G > 2*B
                warm = (cy_arr[:,:,0].astype(int) + cy_arr[:,:,1].astype(int) > 2*cy_arr[:,:,2].astype(int)).mean()
                feats["color_warm_ratio"][i] = float(warm)

        # Visual weight (saliency-weighted)
        feats["visual_weight_x"][i] = (cx_c - 0.5) * area_ratio
        feats["visual_weight_y"][i] = (cy_c - 0.5) * area_ratio

        # cnt_balance: variance of saliency across quadrants of cand
        if crop_v is not None and crop_v.size > 4:
            h_c, w_c = crop_v.shape
            q1 = crop_v[:h_c//2, :w_c//2].mean() if h_c>1 and w_c>1 else 0
            q2 = crop_v[:h_c//2, w_c//2:].mean() if h_c>1 and w_c>1 else 0
            q3 = crop_v[h_c//2:, :w_c//2].mean() if h_c>1 and w_c>1 else 0
            q4 = crop_v[h_c//2:, w_c//2:].mean() if h_c>1 and w_c>1 else 0
            feats["cnt_balance"][i] = float(np.std([q1, q2, q3, q4]))

        # headroom / lead_room / subject_size_ratio
        if mask_np is not None:
            inner = crop_pixel(mask_np, x1, y1, x2, y2, W, H)
            if inner is not None and inner.size > 0:
                # headroom_top = fraction of top 1/4 with no subject
                inner_b = (inner > 64).astype(np.float32)
                if inner_b.shape[0] > 4:
                    top_q = inner_b[:inner_b.shape[0]//4].mean()
                    feats["headroom_top"][i] = float(1.0 - top_q)
                # subject_size_ratio: subject pixels in cand / cand area
                feats["subject_size_ratio"][i] = float(inner_b.sum() / max(inner_b.size, 1))
                # lead_room: side-to-subject distance on V_MIX
                feats["lead_room"][i] = float(max(0, cx_c - 0.5))  # simple proxy

        if (i+1) % 50000 == 0:
            print(f"  features {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    feat_df = pd.DataFrame(feats)
    # Merge metadata
    for c in ["photo_id", "image_path", "src", "x1", "y1", "x2", "y2", "img_w", "img_h", "gt_aspect"]:
        feat_df[c] = df_pool[c].values
    return feat_df
