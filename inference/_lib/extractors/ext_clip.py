"""Per-crop CLIP-L multi-prompt similarity features (faithful port of the benchmark extractor).

For each candidate crop:
  - crop the bbox from the original image (image coords)
  - HF CLIP-L (openai/clip-vit-large-patch14, fp16) image embedding
  - cosine similarity with the 8 reference composition prompts (identical to the cached features)
  - 10 features: 8 clip_sim_{key} + clip_max_sim + clip_top3_mean_sim

Reads invocation scratch/cands.parquet, writes invocation scratch/feat_clip.parquet. Run with the main venv (transformers + CUDA).
"""
from job_contract import extraction_job
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

FLMS_K10 = None
OUT = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared


def load():
    """Load (and memoise) the CLIP-L model once."""
    _shared.get_clip()


@extraction_job({'FLMS_K10': 'cands.parquet', 'OUT': 'feat_clip.parquet'})
def run(args=None):
    t0 = time.time()
    clip_proc, clip_model = _shared.get_clip()
    device = next(clip_model.parameters()).device
    print(f"[device] {device}", flush=True)

    # IDENTICAL prompts + keys to the cached FCDB features
    ref_prompts = [
        "a well-composed professional photograph",
        "a balanced composition with clear subject",
        "leading lines and depth in the composition",
        "rule of thirds composition",
        "symmetrical aesthetic composition",
        "negative space around the subject",
        "a centered subject framing",
        "dynamic off-center composition",
    ]
    prompt_keys = ["pro", "balanced", "leading", "thirds", "symmetric",
                    "negspace", "centered", "dynamic"]
    with torch.no_grad():
        text_in = clip_proc(text=ref_prompts, return_tensors="pt", padding=True).to(device)
        ref_embs = clip_model.get_text_features(**text_in)
        if hasattr(ref_embs, "pooler_output"):
            ref_embs = ref_embs.pooler_output
        ref_embs = ref_embs / (ref_embs.norm(dim=-1, keepdim=True) + 1e-9)
    ref_embs_np = ref_embs.cpu().float().numpy()  # (8, 768)

    df = pd.read_parquet(FLMS_K10)
    print(f"[loaded] {len(df)} cands, {df.photo_id.nunique()} pids", flush=True)

    rows = []
    last_pid = None; last_img = None
    BATCH = 32
    cur_batch = []
    cur_keys = []

    def flush_batch():
        if not cur_batch: return
        with torch.no_grad():
            imgs_in = clip_proc(images=cur_batch, return_tensors="pt").to(device, torch.float16)
            emb = clip_model.get_image_features(**imgs_in)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
            emb_np = emb.cpu().float().numpy()
        for j, (pid, dpp_rank) in enumerate(cur_keys):
            sims = (ref_embs_np @ emb_np[j])
            feats = {f"clip_sim_{k}": float(sims[i]) for i, k in enumerate(prompt_keys)}
            feats["clip_max_sim"] = float(sims.max())
            feats["clip_top3_mean_sim"] = float(np.sort(sims)[-3:].mean())
            rows.append({"photo_id": pid, "dpp_rank": dpp_rank, **feats})

    for i, r in enumerate(df.itertuples(index=False)):
        pid = r.photo_id
        if pid != last_pid:
            try:
                last_img = Image.open(r.image_path).convert("RGB")
                last_pid = pid
            except Exception as exc:
                raise RuntimeError(f"cannot decode candidate image {r.image_path}") from exc
        try:
            cand_crop = last_img.crop((int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
            if cand_crop.size[0] < 8 or cand_crop.size[1] < 8:
                raise ValueError(f"candidate too small: {(pid, r.dpp_rank)}")
            cur_batch.append(cand_crop)
            cur_keys.append((pid, int(r.dpp_rank)))
        except Exception as exc:
            raise RuntimeError(f"cannot extract candidate {(pid, r.dpp_rank)}") from exc
        if len(cur_batch) >= BATCH:
            flush_batch()
            cur_batch = []; cur_keys = []
        if (i+1) % 500 == 0:
            rate = (i+1) / max(1e-6, time.time()-t0)
            eta = (len(df)-i-1) / max(rate, 1e-6)
            print(f"  {i+1}/{len(df)} {rate:.1f}/s ETA {eta:.0f}s", flush=True)
    flush_batch()

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"\n[saved] {OUT} ({len(out)} rows, {time.time()-t0:.0f}s)", flush=True)
    print(out[[c for c in out.columns if c.startswith('clip_')]].describe(), flush=True)
    print(f"[extractor] CLIP-features: CLIP multi-prompt done: {len(out)} candidates ({time.time()-t0:.0f}s)", flush=True)


def main():
    print("[extractor] CLIP-features: CLIP multi-prompt start", flush=True)
    load()
    run()


if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(f"[extractor] CLIP-features: CLIP multi-prompt ERROR: {str(e)[:120]}", flush=True)
        sys.exit(1)
