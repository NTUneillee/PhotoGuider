"""Per-crop SAM DECODER features (faithful port of the benchmark extractor chain).

Stages:
  1. axes 1+2+4 (12 features): SAM single-prompt at the crop center, multi-mask output
  2. axis 3 (5 features): SAM auto-mask everything -> per-crop object statistics
  3. axis 5 (3 features): SAM multi-prompt at the thirds points

Reads invocation scratch/cands.parquet, writes invocation scratch/feat_sam_dec_{stage124,stage3,stage5}.parquet.
Run with sam_env.
"""
from job_contract import extraction_job
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2

sys.stdout.reconfigure(line_buffering=True)

FLMS_K10 = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared

A124_CACHE = None
A3_CACHE = None
A5_CACHE = None


def _bbox_to_mask(bbox, H, W):
    m = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def _mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / max(union, 1))


def _load_flms_k10():
    df = pd.read_parquet(FLMS_K10).sort_values("photo_id").reset_index(drop=True)
    return df


# ─────────── Stage dec.1: axes 1+2+4 ───────────
N_THREADS = 4   # panorama scans hand this extractor 120 independent views; the per-view work is mostly CPU


def _per_photo(df, work):
    """Run ``work(pid, sub)`` for each photo's candidate block (``sub`` in the block's row order) and
    return the rows in the block order of ``df``. Photos are independent, so with several photos
    they run on ``N_THREADS`` threads; a single photo runs inline exactly as before."""
    blocks = [(pid, sub.reset_index(drop=True)) for pid, sub in df.groupby("photo_id", sort=False)]
    if len(blocks) == 1:
        return work(*blocks[0])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(N_THREADS, len(blocks))) as tp:
        results = list(tp.map(lambda b: work(*b), blocks))
    return [row for rows in results for row in rows]


def _read_rgb(path):
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"cannot decode candidate image {path}") from exc


def stage_a124():
    if A124_CACHE.exists():
        print(f"[dec.1] cached: {A124_CACHE.name}", flush=True); return
    print("[dec.1] Single-prompt SAM (axes 1+2+4) on FLMS 5000 cands", flush=True)
    from segment_anything import SamPredictor
    sam = _shared.get_sam()
    df = _load_flms_k10()
    t0 = time.time()

    def work(pid, sub):
        predictor = SamPredictor(sam)
        img = _read_rgb(sub.image_path[0])
        predictor.set_image(img)
        H, W = img.shape[:2]
        rows = []
        for r in sub.itertuples(index=False):
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cand_mask = _bbox_to_mask((x1, y1, x2, y2), H, W)
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[cx, cy]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
            ax1_q_top1 = float(scores.max())
            ax1_q_mean = float(scores.mean())
            ax1_q_std = float(scores.std())
            ax1_q_margin = float(scores.max() - scores.mean())

            areas = masks.reshape(3, -1).sum(axis=1).astype(np.float32)
            ax2_area_cv = float(areas.std() / max(areas.mean(), 1))
            ax2_iou_top12 = float(_mask_iou(masks[0], masks[1]))
            cand_matches = [_mask_iou(masks[k], cand_mask) for k in range(3)]
            ax2_cand_match_max = float(max(cand_matches))
            ax2_match_layer = int(np.argmax(cand_matches))

            best_mask = masks[int(np.argmax(scores))]
            cand_region = img[y1:y2, x1:x2]
            cand_best_mask = best_mask[y1:y2, x1:x2]
            if cand_region.size == 0 or cand_best_mask.sum() < 4 or (~cand_best_mask).sum() < 4:
                ax4_sat_diff = 0.0; ax4_val_diff = 0.0; ax4_lab_dist = 0.0; ax4_fg_var = 0.0
            else:
                hsv = cv2.cvtColor(cand_region, cv2.COLOR_RGB2HSV).astype(np.float32)
                lab = cv2.cvtColor(cand_region, cv2.COLOR_RGB2LAB).astype(np.float32)
                fg_idx = cand_best_mask; bg_idx = ~cand_best_mask
                ax4_sat_diff = float(hsv[..., 1][fg_idx].mean() - hsv[..., 1][bg_idx].mean())
                ax4_val_diff = float(hsv[..., 2][fg_idx].mean() - hsv[..., 2][bg_idx].mean())
                fg_lab_mean = lab[fg_idx].mean(axis=0); bg_lab_mean = lab[bg_idx].mean(axis=0)
                ax4_lab_dist = float(np.linalg.norm(fg_lab_mean - bg_lab_mean))
                ax4_fg_var = float(lab[fg_idx].std(axis=0).mean())

            rows.append({
                "photo_id": pid, "dpp_rank": int(r.dpp_rank),
                "sam_mask_q_top1": ax1_q_top1, "sam_mask_q_mean": ax1_q_mean,
                "sam_mask_q_std": ax1_q_std, "sam_mask_q_margin": ax1_q_margin,
                "sam_h_area_cv": ax2_area_cv, "sam_h_iou_top12": ax2_iou_top12,
                "sam_h_cand_match_max": ax2_cand_match_max, "sam_h_match_layer": ax2_match_layer,
                "sam_fg_bg_sat_diff": ax4_sat_diff, "sam_fg_bg_val_diff": ax4_val_diff,
                "sam_fg_bg_lab_dist": ax4_lab_dist, "sam_fg_var": ax4_fg_var,
            })
        return rows

    rows = _per_photo(df, work)
    pd.DataFrame(rows).to_parquet(A124_CACHE, index=False)
    print(f"[dec.1] saved {A124_CACHE.name} {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[extractor] SAM-decoder: decoder feature groups 1, 2, and 4 done: {len(rows)}", flush=True)


# ─────────── Stage dec.2: axis 3 auto-mask ───────────
def stage_a3():
    if A3_CACHE.exists():
        print(f"[dec.2] cached: {A3_CACHE.name}", flush=True); return
    print("[dec.2] SAM auto-mask FLMS 500 imgs (axis 3)", flush=True)
    from segment_anything import SamAutomaticMaskGenerator
    sam = _shared.get_sam()
    df = _load_flms_k10()
    t0 = time.time()

    def work(pid, cands_here):
        mask_generator = SamAutomaticMaskGenerator(
            sam, points_per_side=16, pred_iou_thresh=0.85,
            stability_score_thresh=0.92, crop_n_layers=0, min_mask_region_area=100,
        )
        img = _read_rgb(cands_here.image_path[0])
        H, W = img.shape[:2]
        with torch.no_grad():
            masks_data = mask_generator.generate(img)
        seg_areas = [(m["segmentation"], int(m["area"])) for m in masks_data]
        rows = []
        for c in cands_here.itertuples(index=False):
            x1, y1, x2, y2 = int(c.x1), int(c.y1), int(c.x2), int(c.y2)
            cand_area = max((x2-x1)*(y2-y1), 1)
            in_cand_areas = []
            for seg, area_full in seg_areas:
                if seg.shape[0] != H or seg.shape[1] != W: continue
                inter = int(seg[y1:y2, x1:x2].sum())
                if inter / max(area_full, 1) > 0.5:
                    in_cand_areas.append(inter)
            n = len(in_cand_areas)
            if n > 0:
                arr = np.array(in_cand_areas, dtype=np.float32)
                arr_sorted = np.sort(arr)[::-1]
                top1 = float(arr_sorted[0])
                top3 = float(arr_sorted[:3].sum())
                obj_cv = float(arr.std() / max(arr.mean(), 1))
                dom_ratio = float(top1 / max(arr.sum(), 1))
            else:
                top1 = top3 = obj_cv = dom_ratio = 0.0
            rows.append({
                "photo_id": pid, "dpp_rank": int(c.dpp_rank),
                "sam_obj_count_in_cand": n,
                "sam_obj_top1_share": top1 / cand_area,
                "sam_obj_top3_share": top3 / cand_area,
                "sam_obj_area_cv": obj_cv,
                "sam_dominant_ratio": dom_ratio,
            })
        return rows

    rows = _per_photo(df, work)
    pd.DataFrame(rows).to_parquet(A3_CACHE, index=False)
    print(f"[dec.2] saved {A3_CACHE.name} {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[extractor] SAM-decoder: decoder feature group 3 done: {len(rows)}", flush=True)


# ─────────── Stage dec.3: axis 5 multi-prompt thirds ───────────
def stage_a5():
    if A5_CACHE.exists():
        print(f"[dec.3] cached: {A5_CACHE.name}", flush=True); return
    print("[dec.3] Multi-prompt SAM at thirds (axis 5)", flush=True)
    from segment_anything import SamPredictor
    sam = _shared.get_sam()

    a124 = pd.read_parquet(A124_CACHE)
    a124_map = a124.set_index(["photo_id", "dpp_rank"])["sam_mask_q_top1"].to_dict()

    df = _load_flms_k10()
    t0 = time.time()

    def work(pid, sub):
        predictor = SamPredictor(sam)
        predictor.set_image(_read_rgb(sub.image_path[0]))
        rows = []
        for r in sub.itertuples(index=False):
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cw, ch = x2-x1, y2-y1
            thirds_pts = [
                (x1 + cw//3, y1 + ch//3),
                (x1 + 2*cw//3, y1 + ch//3),
                (x1 + cw//3, y1 + 2*ch//3),
                (x1 + 2*cw//3, y1 + 2*ch//3),
            ]
            per_pt_q = []
            with torch.no_grad():
                for pt in thirds_pts:
                    _, scores, _ = predictor.predict(
                        point_coords=np.array([pt]), point_labels=np.array([1]),
                        multimask_output=True,
                    )
                    per_pt_q.append(float(scores.max()))
            thirds_q_mean = float(np.mean(per_pt_q))
            thirds_q_max = float(np.max(per_pt_q))
            center_q = a124_map.get((pid, int(r.dpp_rank)), thirds_q_mean)
            thirds_vs_center = thirds_q_mean - center_q
            rows.append({
                "photo_id": pid, "dpp_rank": int(r.dpp_rank),
                "sam_thirds_q_mean": thirds_q_mean,
                "sam_thirds_q_max": thirds_q_max,
                "sam_thirds_vs_center": thirds_vs_center,
            })
        return rows

    rows = _per_photo(df, work)
    pd.DataFrame(rows).to_parquet(A5_CACHE, index=False)
    print(f"[dec.3] saved {A5_CACHE.name} {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[extractor] SAM-decoder: decoder feature group 5 done: {len(rows)}", flush=True)


def load():
    _shared.get_sam()


@extraction_job({'FLMS_K10': 'cands.parquet', 'A124_CACHE': 'feat_sam_dec_stage124.parquet', 'A3_CACHE': 'feat_sam_dec_stage3.parquet', 'A5_CACHE': 'feat_sam_dec_stage5.parquet'})
def run(args=None):
    stage_a124()
    stage_a3()
    stage_a5()


def main():
    print("[extractor] SAM-decoder: SAM decoder feature chain start", flush=True)
    t0 = time.time()
    load()
    run()
    elapsed = (time.time()-t0)/60
    msg = f"SAM decoder features (3 axes) done ({elapsed:.1f}min)"
    print(f"[extractor] SAM-decoder: {msg}", flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(f"[extractor] SAM-decoder: SAM decoder ERROR: {str(e)[:120]}", flush=True)
        sys.exit(1)
