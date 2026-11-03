"""Per-crop LAION aesthetic score (laion_env). Reads invocation scratch/cands.parquet, writes invocation scratch/feat_laion.parquet."""
from job_contract import extraction_job
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)
CANDS = None
OUT = None
BATCH = 64
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared


def load():
    _shared.get_laion()


@extraction_job({'CANDS': 'cands.parquet', 'OUT': 'feat_laion.parquet'})
def run(args=None):
    clip_model, preprocess, head, DEV = _shared.get_laion()
    df = pd.read_parquet(CANDS); print(f"{len(df)} cands", flush=True)
    aes = np.zeros(len(df), dtype=np.float32)
    cache = {}; buf_imgs, buf_idx = [], []
    def flush():
        if not buf_imgs: return
        x = torch.stack(buf_imgs).to(DEV)
        with torch.no_grad():
            f = clip_model.encode_image(x); f = f/f.norm(dim=-1, keepdim=True)
            a = head(f.float()).squeeze(-1).cpu().numpy()
        for j, idx in enumerate(buf_idx): aes[idx] = a[j]
        buf_imgs.clear(); buf_idx.clear()
    for i, r in enumerate(df.itertuples(index=False)):
        if r.image_path not in cache:
            cache[r.image_path] = Image.open(r.image_path).convert("RGB")
            if len(cache) > 8: cache.pop(next(iter(cache)))
        img = cache[r.image_path]; Wi, Hi = img.size
        ix1, iy1, ix2, iy2 = _shared.scaled_box(r.x1, r.y1, r.x2, r.y2, r.img_w, r.img_h, Wi, Hi)
        buf_imgs.append(preprocess(img.crop((ix1, iy1, ix2, iy2)))); buf_idx.append(i)
        if len(buf_imgs) >= BATCH: flush()
        if (i+1) % 4000 == 0: print(f"  {i+1}/{len(df)}", flush=True)
    flush()
    df["aes"] = aes; df.to_parquet(OUT); print(f"saved -> {OUT}")


def main():
    load(); run()


if __name__ == "__main__": main()
