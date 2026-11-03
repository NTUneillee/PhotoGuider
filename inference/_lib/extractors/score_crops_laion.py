"""Score refine scaled-crops with LAION aesthetic (laion_env).
Reads invocation scratch/refine_crops.parquet (image_path,img_w,img_h,x1,y1,x2,y2,s),
writes invocation scratch/refine_laion.parquet ([s, score]). Same crop+head as ext_laion."""
from job_contract import extraction_job
import sys
from pathlib import Path
import pandas as pd, torch
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)
IN = None; OUT = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared


def load():
    _shared.get_laion()


@extraction_job({'IN': 'refine_crops.parquet', 'OUT': 'refine_laion.parquet'})
def run(args=None):
    clip_model, preprocess, head, DEV = _shared.get_laion()
    df = pd.read_parquet(IN)
    img = Image.open(df.image_path.iloc[0]).convert("RGB"); Wi, Hi = img.size
    scores = []
    for r in df.itertuples(index=False):
        ix1, iy1, ix2, iy2 = _shared.scaled_box(r.x1, r.y1, r.x2, r.y2, r.img_w, r.img_h, Wi, Hi)
        x = preprocess(img.crop((ix1, iy1, ix2, iy2))).unsqueeze(0).to(DEV)
        with torch.no_grad():
            f = clip_model.encode_image(x); f = f/f.norm(dim=-1, keepdim=True)
            scores.append(float(head(f.float()).squeeze().cpu().numpy()))
    pd.DataFrame({"s": df.s.values, "score": scores}).to_parquet(OUT)
    print(f"saved -> {OUT}")


def main():
    load(); run()


if __name__ == "__main__": main()
