"""Per-crop FGAesQ fine-grained aesthetic score (faithful port of the benchmark extractor).

For each FLMS K=10 cand:
  - Crop bbox from original image
  - DiffToken process_image (multi-scale patches + pos_embed + mask)
  - FGAesQ.forward → distribution logits → expectation score in [1, 10]

Reads invocation scratch/cands.parquet, writes invocation scratch/feat_fgaesq.parquet.

Run with: envs/fgaesq/venv (has OpenAI clip + torch)
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
BATCH = 32
# DiffToken's upstream random patch selection is unseeded; seeding before model init (in
# _shared.get_fgaesq) and again before each processing+forward batch makes extraction
# repeat-deterministic.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared
# The shared loader adds the configured FGAesQ source directory.


def load():
    """Load (and memoise) FGAesQ + DiffToken once (seeded init preserved)."""
    _shared.get_fgaesq()


@extraction_job({'FLMS_K10': 'cands.parquet', 'OUT': 'feat_fgaesq.parquet'})
def run(args=None):
    t0 = time.time()
    model, proc, dev = _shared.get_fgaesq()
    print(f"[device] {dev}", flush=True)

    df = pd.read_parquet(FLMS_K10).reset_index(drop=True).copy()
    df["_gidx"] = np.arange(len(df))
    print(f"[loaded] {len(df)} FLMS cands, {df.photo_id.nunique()} pids", flush=True)

    df_sorted = df.sort_values(["image_path", "_gidx"]).reset_index(drop=True)

    last_path = None; last_img = None
    def crop_item(i):
        nonlocal last_path, last_img
        r = df_sorted.iloc[i]
        if r["image_path"] != last_path:
            last_img = Image.open(r["image_path"]).convert("RGB")
            last_path = r["image_path"]
        crop = last_img.crop((int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
        p, e, m = proc.process_image(crop, is_train=False)
        return {"p": p, "e": e, "m": m, "gidx": int(r._gidx)}

    scores = np.full(len(df), np.nan, dtype=np.float32)
    n = len(df_sorted)
    last = time.time()
    with torch.no_grad():
        for start in range(0, n, BATCH):
            _shared._seed_all()      # one seeded window per batch: processing + forward
            items = [crop_item(i) for i in range(start, min(start + BATCH, n))]
            p = torch.stack([it["p"] for it in items]).to(dev, non_blocking=True)
            e = torch.stack([it["e"] for it in items]).to(dev, non_blocking=True)
            m = torch.stack([it["m"] for it in items]).to(dev, non_blocking=True)
            out = model(p, e, m).squeeze(-1).cpu().numpy().reshape(-1)
            for it, s in zip(items, out.tolist()):
                scores[it["gidx"]] = s
            done = min(start + BATCH, n)
            if time.time() - last >= 20 or done >= n:
                rate = done / max(1e-6, time.time()-t0)
                eta = (n-done) / max(rate, 1e-6)
                print(f"  [FGAesQ] {done}/{n} {rate:.1f}/s ETA {eta:.0f}s", flush=True)
                last = time.time()

    df_out = pd.DataFrame({
        "photo_id": df["photo_id"].values,
        "dpp_rank": df["dpp_rank"].values.astype(int),
        "fgaesq_score": scores,
    })
    # match the cached-feature derivatives (score + rank_K10 + pct_K10) for safety
    df_out["fgaesq_rank_K10"] = (
        df_out.groupby("photo_id")["fgaesq_score"]
              .rank(method="min", ascending=False).astype(int) - 1
    )
    def pmm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)
    df_out["fgaesq_pct_K10"] = df_out.groupby("photo_id")["fgaesq_score"].transform(pmm)
    df_out.to_parquet(OUT, index=False)
    print(f"\n[saved] {OUT} ({len(df_out)} rows, {time.time()-t0:.0f}s)", flush=True)
    print(df_out[["fgaesq_score"]].describe(), flush=True)
    print(f"[extractor] aesthetic-features: FGAesQ done: {len(df_out)} candidates ({time.time()-t0:.0f}s)", flush=True)


def main():
    print("[extractor] aesthetic-features: FGAesQ start", flush=True)
    print("[loading FGAesQ]", flush=True)
    load()
    run()


if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(f"[extractor] aesthetic-features: FGAesQ ERROR: {str(e)[:120]}", flush=True)
        sys.exit(1)
