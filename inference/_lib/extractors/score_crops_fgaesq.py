"""Score refine scaled-crops with FGAesQ (envs/fgaesq/venv).
Reads invocation scratch/refine_crops.parquet, writes invocation scratch/refine_fgaesq.parquet ([s, score]).
Same DiffToken + FGAesQ.forward as ext_fgaesq."""
from job_contract import extraction_job
import sys, traceback
from pathlib import Path
import pandas as pd, torch
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)
IN = None; OUT = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared


def load():
    _shared.get_fgaesq()


@extraction_job({'IN': 'refine_crops.parquet', 'OUT': 'refine_fgaesq.parquet'})
def run(args=None):
    model, proc, dev = _shared.get_fgaesq()
    df = pd.read_parquet(IN)
    img = Image.open(df.image_path.iloc[0]).convert("RGB")
    scores = []
    with torch.no_grad():
        for r in df.itertuples(index=False):
            _shared._seed_all()      # one seeded window per crop: processing + forward
            crop = img.crop((int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
            p, e, m = proc.process_image(crop, is_train=False)
            out = model(p.unsqueeze(0).to(dev), e.unsqueeze(0).to(dev), m.unsqueeze(0).to(dev))
            scores.append(float(out.squeeze().cpu().numpy()))
    pd.DataFrame({"s": df.s.values, "score": scores}).to_parquet(OUT)
    print(f"saved -> {OUT}")


def main():
    load(); run()


if __name__ == "__main__":
    try: main()
    except Exception:
        print(traceback.format_exc(), flush=True); sys.exit(1)
