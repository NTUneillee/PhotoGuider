"""Per-crop SAM ENCODER features (faithful port of the benchmark extractor chain).

Stages:
  1. SAM full-image embeddings (256-d per image)
  2. SAM candidate sub-image embeddings (256-d per crop)
  A. hand-crafted scalars (5 features): sam_emb_norm/var/kurt/cos_to_image/pid_norm_rank
  B. image patch norms (64x64) -> 5 per-crop spatial scalars
  C. point-prompt SAM mask features (5 per crop)

Reads invocation scratch/cands.parquet, writes invocation scratch/feat_sam_enc_{handcrafted,spatial,mask}.parquet.
Run with sam_env.
"""
from job_contract import extraction_job
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from segment_anything import SamPredictor

sys.stdout.reconfigure(line_buffering=True)

FLMS_K10 = None
# The live InSPyReNet subject mask is computed in the main process (feat_saliency) and consumed
# there by every other mask feature. This extractor runs in its own environment, so the one feature
# here that compares the SAM mask against the subject mask (sam_mask_vs_inspy_iou) receives that
# same live mask through the job scratch (subject_masks.pkl, written by Extractors.deep). No cache
# is read; a missing file yields 0.
SUBJECT_MASKS = None   # bound per job to scratch/subject_masks.pkl
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared

IMG_EMB = None
CAND_EMB = None
IMG_PATCH = None
A_FEATS = None
B_FEATS = None
C_FEATS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4
SAM_INPUT = 1024
EMB_DIM = 256

SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def preprocess_for_sam(pil_img):
    img = pil_img.resize((SAM_INPUT, SAM_INPUT), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _encode_oom_safe(sam_model, tensors, reduce):
    """Forward a batch through the SAM image encoder; on a transient CUDA OOM (the batch
    forward is this chain's peak-memory op and the GPU is shared with the other resident
    extractor workers), free the allocator cache and retry one item at a time. Per-sample
    outputs are batch-size independent, so the fallback is numerically equivalent."""
    def fwd(ts):
        x = torch.stack(ts).to(DEVICE)
        x = (x - SAM_PIXEL_MEAN.to(DEVICE)) / SAM_PIXEL_STD.to(DEVICE)
        with torch.no_grad():
            feat = sam_model.image_encoder(x)  # (B, 256, 64, 64)
        return reduce(feat).float().cpu().numpy()
    try:
        return fwd(tensors)
    except torch.cuda.OutOfMemoryError:
        log(f"  CUDA OOM on batch of {len(tensors)} -> empty_cache + per-item retry")
        torch.cuda.empty_cache()
        return np.concatenate([fwd(tensors[i:i+1]) for i in range(len(tensors))])


def encode_batch_mean(sam_model, tensors):
    """Mean-pool SAM (256, 64, 64) feature → (256,)."""
    return _encode_oom_safe(sam_model, tensors, lambda f: f.mean(dim=(2, 3)))


def encode_batch_patch_norms(sam_model, tensors):
    """Per-patch L2 norm of SAM (B, 256, 64, 64) → (B, 64, 64)."""
    return _encode_oom_safe(sam_model, tensors, lambda f: f.norm(dim=1))


# ─────────── Stage enc.1: full-image embeddings ───────────
def stage_44_1_image_emb(sam_model):
    if IMG_EMB.exists():
        log("[enc.1] image emb exists, skip"); return
    log("=" * 60); log("[enc.1] FLMS 500 full-image embeddings"); log("=" * 60)
    df = pd.read_parquet(FLMS_K10).sort_values("photo_id").reset_index(drop=True)
    pid_to_img = {p: g.image_path.iloc[0] for p, g in df.groupby("photo_id")}
    pids_sorted = sorted(pid_to_img.keys())
    items = []
    for pid in pids_sorted:
        try: items.append((pid, Image.open(pid_to_img[pid]).convert("RGB")))
        except Exception as e: log(f"  pid {pid} fail: {e}")
    log(f"  prepared {len(items)} imgs")

    out_idxs = []; out_embs = []
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i+BATCH]
        idxs = [c[0] for c in chunk]
        tensors = [preprocess_for_sam(c[1]) for c in chunk]
        emb = encode_batch_mean(sam_model, tensors)
        out_idxs.extend(idxs); out_embs.append(emb)
        if (i // BATCH) % 10 == 0:
            log(f"  {i+len(chunk)}/{len(items)} ({time.time()-t0:.0f}s)")
    embs = np.concatenate(out_embs)
    df_out = pd.DataFrame(embs, columns=[f"e{j}" for j in range(EMB_DIM)])
    df_out.insert(0, "photo_id", out_idxs)
    df_out.to_parquet(IMG_EMB, index=False)
    log(f"  saved {IMG_EMB.name} {df_out.shape}")


# ─────────── Stage enc.2: cand sub-image embeddings ───────────
def stage_44_2_cand_emb(sam_model):
    if CAND_EMB.exists():
        log("[enc.2] cand emb exists, skip"); return
    log("=" * 60); log("[enc.2] FLMS 5000 cand sub-image embeddings"); log("=" * 60)
    df = pd.read_parquet(FLMS_K10).sort_values("photo_id").reset_index(drop=True)
    pid_to_img = {p: g.image_path.iloc[0] for p, g in df.groupby("photo_id")}
    pid_to_wh = {p: (float(g.img_w.iloc[0]), float(g.img_h.iloc[0]))
                  for p, g in df.groupby("photo_id")}

    img_cache = {}
    items = []
    for i, row in enumerate(df.itertuples(index=False)):
        pid = row.photo_id
        if pid not in img_cache:
            img_cache[pid] = Image.open(pid_to_img[pid]).convert("RGB")
            if len(img_cache) > 8:
                oldest = next(iter(img_cache))
                if oldest != pid: del img_cache[oldest]
        img = img_cache[pid]
        W_img, H_img = img.size
        ref_W, ref_H = pid_to_wh[pid]
        ix1, iy1, ix2, iy2 = _shared.scaled_box(row.x1, row.y1, row.x2, row.y2, ref_W, ref_H, W_img, H_img)
        cropped = img.crop((ix1, iy1, ix2, iy2))
        items.append(((pid, int(row.dpp_rank)), cropped))

    log(f"  prepared {len(items)} cand crops, encoding ...")
    out_idxs = []; out_embs = []
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i+BATCH]
        idxs = [c[0] for c in chunk]
        tensors = [preprocess_for_sam(c[1]) for c in chunk]
        emb = encode_batch_mean(sam_model, tensors)
        out_idxs.extend(idxs); out_embs.append(emb)
        if (i // BATCH) % 50 == 0:
            log(f"  {i+len(chunk)}/{len(items)} ({time.time()-t0:.0f}s)")
    embs = np.concatenate(out_embs)
    df_out = pd.DataFrame(embs, columns=[f"e{j}" for j in range(EMB_DIM)])
    df_out.insert(0, "dpp_rank", [x[1] for x in out_idxs])
    df_out.insert(0, "photo_id", [x[0] for x in out_idxs])
    df_out.to_parquet(CAND_EMB, index=False)
    log(f"  saved {CAND_EMB.name} {df_out.shape}")


# ─────────── Stage enc.A: hand-crafted scalars ───────────
def stage_46_A():
    if A_FEATS.exists():
        log("[enc.A] features exist, skip"); return
    log("=" * 60); log("[enc.A] Hand-crafted scalars from SAM cand emb"); log("=" * 60)
    df_cand = pd.read_parquet(CAND_EMB)
    df_img = pd.read_parquet(IMG_EMB)
    e_cols = [c for c in df_cand.columns if c.startswith("e")]
    cand_e = df_cand[e_cols].values.astype(np.float32)
    img_e_lookup = {r.photo_id: np.array([getattr(r, c) for c in e_cols], dtype=np.float32)
                    for r in df_img.itertuples(index=False)}

    cand_norm = np.linalg.norm(cand_e, axis=1)
    cand_var = cand_e.std(axis=1)
    cand_kurt = stats.kurtosis(cand_e, axis=1)
    cand_e_norm = cand_e / (cand_norm[:, None] + 1e-9)
    cos_to_img = np.zeros(len(df_cand))
    for i, pid in enumerate(df_cand["photo_id"].values):
        img_e = img_e_lookup[pid]
        img_e_n = img_e / (np.linalg.norm(img_e) + 1e-9)
        cos_to_img[i] = float(np.dot(cand_e_norm[i], img_e_n))

    df_out = pd.DataFrame({
        "photo_id": df_cand["photo_id"].values,
        "dpp_rank": df_cand["dpp_rank"].values,
        "sam_emb_norm": cand_norm.astype(np.float32),
        "sam_emb_var": cand_var.astype(np.float32),
        "sam_emb_kurt": cand_kurt.astype(np.float32),
        "sam_cos_to_image": cos_to_img.astype(np.float32),
    })
    df_out["sam_pid_norm_rank"] = (
        df_out.groupby("photo_id")["sam_emb_norm"]
              .rank(method="min", ascending=False).astype(int) - 1
    )
    df_out.to_parquet(A_FEATS, index=False)
    log(f"  saved {A_FEATS.name}")


# ─────────── Stage enc.B-encode: image patch norms ───────────
def stage_46_B_encode(sam_model):
    if IMG_PATCH.exists():
        log("[enc.B-enc] patch norms cache exists, skip"); return
    log("=" * 60); log("[enc.B-enc] FLMS 500 imgs → 64×64 patch norms"); log("=" * 60)
    df = pd.read_parquet(FLMS_K10).sort_values("photo_id").reset_index(drop=True)
    pid_to_img = {p: g.image_path.iloc[0] for p, g in df.groupby("photo_id")}
    pids_sorted = sorted(pid_to_img.keys())

    out_rows = []; items = []
    t0 = time.time()
    for idx, pid in enumerate(pids_sorted):
        try:
            items.append((pid, preprocess_for_sam(Image.open(pid_to_img[pid]).convert("RGB"))))
        except Exception as e:
            log(f"  pid {pid} fail: {e}"); continue
        if len(items) >= BATCH:
            patch_norms = encode_batch_patch_norms(sam_model, [item[1] for item in items])
            for k, (p, _) in enumerate(items):
                out_rows.append({"photo_id": p, "patch_norms_flat": patch_norms[k].flatten()})
            items = []
            if (idx // 4) % 25 == 0:
                log(f"  {idx+1}/{len(pids_sorted)} ({time.time()-t0:.0f}s)")
    if items:
        patch_norms = encode_batch_patch_norms(sam_model, [item[1] for item in items])
        for k, (p, _) in enumerate(items):
            out_rows.append({"photo_id": p, "patch_norms_flat": patch_norms[k].flatten()})

    pids = [r["photo_id"] for r in out_rows]
    patch_mat = np.stack([r["patch_norms_flat"] for r in out_rows])
    df_save = pd.DataFrame(patch_mat, columns=[f"p{i}" for i in range(patch_mat.shape[1])])
    df_save.insert(0, "photo_id", pids)
    df_save.to_parquet(IMG_PATCH, index=False)
    log(f"  saved {IMG_PATCH.name} {df_save.shape}")


# ─────────── Stage enc.B-compute: per-cand spatial scalars ───────────
def stage_46_B_compute():
    if B_FEATS.exists():
        log("[enc.B-comp] features exist, skip"); return
    log("=" * 60); log("[enc.B-comp] Per-cand spatial scalars"); log("=" * 60)
    df_patch = pd.read_parquet(IMG_PATCH)
    p_cols = [c for c in df_patch.columns if c.startswith("p") and c != "photo_id"]
    patch_mat = df_patch[p_cols].values.reshape(len(df_patch), 64, 64).astype(np.float32)
    pid_to_idx = {r.photo_id: i for i, r in enumerate(df_patch.itertuples(index=False))}

    df_cands = pd.read_parquet(FLMS_K10).sort_values("photo_id").reset_index(drop=True)
    out_rows = []
    for row in df_cands.itertuples(index=False):
        pid = row.photo_id
        if pid not in pid_to_idx:
            out_rows.append((pid, int(row.dpp_rank), 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        patch_norms = patch_mat[pid_to_idx[pid]]
        ref_W, ref_H = float(row.img_w), float(row.img_h)
        fx1 = max(0, min(int(round(row.x1 / ref_W * 64)), 63))
        fy1 = max(0, min(int(round(row.y1 / ref_H * 64)), 63))
        fx2 = max(fx1 + 1, min(int(round(row.x2 / ref_W * 64)), 64))
        fy2 = max(fy1 + 1, min(int(round(row.y2 / ref_H * 64)), 64))
        region = patch_norms[fy1:fy2, fx1:fx2]
        mask = np.zeros_like(patch_norms, dtype=bool)
        mask[fy1:fy2, fx1:fx2] = True
        outside = patch_norms[~mask]
        # a full-frame box has no outside region, so all four candidate statistics are zeroed, not
        # only the ratio; every panorama view is scored this way, matching the shipped models.
        if region.size == 0 or outside.size == 0:
            out_rows.append((pid, int(row.dpp_rank), 0.0, 0.0, 0.0, 0.0, float(patch_norms.sum())))
            continue
        cand_mean = float(region.mean()); cand_max = float(region.max())
        cand_std = float(region.std())
        outside_mean = float(outside.mean())
        cand_to_out = cand_mean / (outside_mean + 1e-9)
        img_total = float(patch_norms.sum())
        out_rows.append((pid, int(row.dpp_rank), cand_mean, cand_max, cand_std,
                          cand_to_out, img_total))
    df_out = pd.DataFrame(out_rows, columns=[
        "photo_id", "dpp_rank",
        "sam_cand_norm_mean", "sam_cand_norm_max", "sam_cand_norm_std",
        "sam_cand_to_outside_ratio", "sam_image_total_norm",
    ])
    df_out.to_parquet(B_FEATS, index=False)
    log(f"  saved {B_FEATS.name}")


# ─────────── Stage enc.C: point-prompt mask features ───────────
def stage_46_C_lite(sam_model):
    if C_FEATS.exists():
        log("[enc.C-lite] mask features exist, skip"); return
    log("=" * 60); log("[enc.C-lite] Point-prompt SAM mask features"); log("=" * 60)
    predictor = SamPredictor(sam_model)
    masks_inspy = pickle.load(open(SUBJECT_MASKS, "rb")) if SUBJECT_MASKS and Path(SUBJECT_MASKS).exists() else {}
    log(f"subject masks for {len(masks_inspy)} images" if masks_inspy else "no subject masks -> sam_mask_vs_inspy_iou = 0")

    df_cands = pd.read_parquet(FLMS_K10).sort_values("photo_id").reset_index(drop=True)
    pid_to_img = {p: g.image_path.iloc[0] for p, g in df_cands.groupby("photo_id")}

    out_rows = []
    img_cache = {}
    t0 = time.time(); last = t0
    current_pid = None
    for i, row in enumerate(df_cands.itertuples(index=False)):
        pid = row.photo_id
        if pid != current_pid:
            if pid not in img_cache:
                try:
                    img = Image.open(pid_to_img[pid]).convert("RGB")
                    img_cache[pid] = np.array(img)
                    if len(img_cache) > 4:
                        oldest = next(iter(img_cache))
                        if oldest != pid: del img_cache[oldest]
                except Exception as e:
                    log(f"  pid {pid} fail: {e}")
                    out_rows.append((pid, int(row.dpp_rank), 0.0, 0.0, 0.0, 0.0, 0.0))
                    continue
            predictor.set_image(img_cache[pid])
            current_pid = pid

        H_img, W_img = img_cache[pid].shape[0], img_cache[pid].shape[1]
        ref_W, ref_H = float(row.img_w), float(row.img_h)
        x1, y1, x2, y2 = _shared.scaled_box(row.x1, row.y1, row.x2, row.y2, ref_W, ref_H, W_img, H_img)
        cand_area = (x2 - x1) * (y2 - y1)
        cand_cx = (x1 + x2) / 2
        cand_cy = (y1 + y2) / 2
        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cand_cx, cand_cy]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            mask = masks[0].astype(bool)
        except Exception as exc:
            raise RuntimeError(f"SAM mask extraction failed for photo {pid} rank {int(row.dpp_rank)}") from exc

        cand_mask = mask[y1:y2, x1:x2]
        mask_area = float(cand_mask.sum())
        sam_mask_area_ratio = mask_area / max(cand_area, 1)
        if mask_area >= 1:
            ys, xs = np.where(cand_mask)
            mask_cx = xs.mean() + x1; mask_cy = ys.mean() + y1
            sam_mask_centered_x = abs(mask_cx - cand_cx) / max(x2 - x1, 1)
            sam_mask_centered_y = abs(mask_cy - cand_cy) / max(y2 - y1, 1)
            top_touch = cand_mask[0, :].sum()
            bot_touch = cand_mask[-1, :].sum() if cand_mask.shape[0] > 1 else 0
            left_touch = cand_mask[:, 0].sum()
            right_touch = cand_mask[:, -1].sum() if cand_mask.shape[1] > 1 else 0
            boundary_pixels = top_touch + bot_touch + left_touch + right_touch
            sam_mask_boundary_touch = float(boundary_pixels) / max(mask_area, 1)
        else:
            sam_mask_centered_x = 0.5; sam_mask_centered_y = 0.5
            sam_mask_boundary_touch = 0.0

        sam_mask_vs_inspy_iou = 0.0
        if pid in masks_inspy:
            inspy_mask = masks_inspy[pid][2]
            inspy_resized = np.array(Image.fromarray(inspy_mask).resize(
                (W_img, H_img), Image.NEAREST)) >= 64
            inspy_cand = inspy_resized[y1:y2, x1:x2]
            inter = (cand_mask & inspy_cand).sum()
            union = (cand_mask | inspy_cand).sum()
            sam_mask_vs_inspy_iou = float(inter / max(union, 1))

        out_rows.append((pid, int(row.dpp_rank), sam_mask_area_ratio,
                          sam_mask_centered_x, sam_mask_centered_y,
                          sam_mask_boundary_touch, sam_mask_vs_inspy_iou))

        if time.time() - last > 30:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(df_cands) - i - 1) / rate / 60
            log(f"  {i+1}/{len(df_cands)} cands ({eta:.1f} min ETA)")
            last = time.time()

    df_out = pd.DataFrame(out_rows, columns=[
        "photo_id", "dpp_rank",
        "sam_mask_area_ratio", "sam_mask_centered_x", "sam_mask_centered_y",
        "sam_mask_boundary_touch", "sam_mask_vs_inspy_iou",
    ])
    df_out.to_parquet(C_FEATS, index=False)
    log(f"  saved {C_FEATS.name}")


# ─────────── Main ───────────
def load():
    _shared.get_sam()


@extraction_job({'FLMS_K10': 'cands.parquet', 'SUBJECT_MASKS': 'subject_masks.pkl', 'IMG_EMB': 'feat_sam_image_embeddings.parquet', 'CAND_EMB': 'feat_sam_cand_embeddings.parquet', 'IMG_PATCH': 'feat_sam_image_patch_norms.parquet', 'A_FEATS': 'feat_sam_enc_handcrafted.parquet', 'B_FEATS': 'feat_sam_enc_spatial.parquet', 'C_FEATS': 'feat_sam_enc_mask.parquet'})
def run(args=None):
    sam_model = _shared.get_sam()
    stage_44_1_image_emb(sam_model)
    stage_44_2_cand_emb(sam_model)
    stage_46_A()
    stage_46_B_encode(sam_model)
    stage_46_B_compute()
    stage_46_C_lite(sam_model)


def main():
    print("[extractor] SAM-encoder: SAM encoder feature chain start", flush=True)
    load()
    run()
    elapsed = (time.time() - T0) / 60
    msg = f"SAM encoder features done ({elapsed:.1f}min): A+B+C-lite all cached"
    log(msg)
    print(f"[extractor] SAM-encoder: {msg}", flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(f"[extractor] SAM-encoder: SAM encoder ERROR: {str(e)[:120]}", flush=True)
        sys.exit(1)
