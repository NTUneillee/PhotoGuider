"""Score rendered views with the live 60-feature pipeline and the shipped composition scorer.

  phase_maps(rows)                 live DINOv3 saliency + InSPyReNet subject mask per view (models resident)
  phase_deep(rows, maps, tmp, ..)  whole-frame box [0,0,W,H] per view -> core + deep 60 features
  score_60f(build_scorer(), frame) additive view scorer (token-grouped NAM, GAICD-calibrated) -> raw score, MOS

Reuses the inference machinery (feature_extractors.Extractors, consultant_features core features).
The 60 feature names are identical to the scorer's feature list (asserted at load).
"""
import gc
import importlib.util
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                               # PanoCropping/
RELEASE = ROOT.parent                            # release root (holds inference/)
INF_DIR = RELEASE / "inference"


def imp(n, p):
    m = sys.modules.get(n)                        # already imported (the web app loads photoguider first): reuse it
    if m is not None and Path(getattr(m, "__file__", "")).resolve() == Path(p).resolve():
        return m
    s = importlib.util.spec_from_file_location(n, str(p))
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


INFER = imp("infer", INF_DIR / "infer.py")
FS = imp("feat_saliency", INF_DIR / "_lib" / "feat_saliency.py")
M = INFER.M                                   # pipeline constants (M.FC = 60F names)
M02 = INFER.M02                               # core-feature extractor

sys.path.insert(0, str(ROOT / "models"))
import scorer as SC                            # noqa: E402  (self-contained scorer loader)

MODEL_A = SC.load()
FEATS = MODEL_A["feats"]                        # 60F, == M.FC (asserted below)
assert set(FEATS) == set(M.FC), "scorer 60F != pipeline M.FC"


def build_scorer():
    return SC.build_scorer(MODEL_A)


def score_60f(model, df):
    sc, mos = SC.score_60f(model, MODEL_A, df)
    return sc.astype(np.float64), np.asarray(mos, np.float64)


# ── Phase A: saliency + mask maps per view (models resident) ──
def phase_maps(rows):
    t0 = time.time()
    maps = {}
    for i, r in enumerate(rows):
        vid = r["vid"]
        try:
            vmix, v1 = FS.build_saliency(r["path"])
            mask = FS.build_mask(r["path"])
            maps[vid] = dict(vmix=vmix, v1=v1, mask=mask)
        except Exception as e:
            print(f"[maps] view {vid} FAILED: {e}", flush=True)
            maps[vid] = None
        if (i + 1) % 30 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            print(f"[maps] {i+1}/{len(rows)} ({el/(i+1):.2f}s/view)", flush=True)
    # free saliency GPU before the deep extractor workers run
    try:
        FS._STATE.clear()
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
    return maps


def map_triple(maps):
    """The (vmix, v1, mask) dicts keyed by vid that the core featurizer takes."""
    return ({k: (None, None, v["vmix"]) for k, v in maps.items() if v},
            {k: (None, None, v["v1"]) for k, v in maps.items() if v},
            {k: (None, None, v["mask"]) for k, v in maps.items() if v})


# ── Phase B: core + deep 60F over the whole grid in one batch ──
def phase_deep(rows, maps, tmp, runner=None):
    """Core + deep 60F for every view, extracted in ``tmp`` (served by ``runner`` when a pool is up)."""
    missing = [r["vid"] for r in rows if maps.get(r["vid"]) is None]
    if missing:
        raise RuntimeError(f"view map extraction failed for candidate views: {missing}")
    from _lib.feature_contract import whole_view_rows
    man = pd.DataFrame(whole_view_rows(rows))
    core = M02.stage_c_features(man.reset_index(drop=True), map_triple(maps))
    frame = pd.concat([man.reset_index(drop=True), core.reset_index(drop=True)], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = INFER.Extractors(tmp, runner).deep(frame, subject_masks=map_triple(maps)[2])   # batched CLIP/LAION/FGAesQ/SAM over all views
    frame = INFER.with_pi(frame)               # laion_pi_minmax
    return frame
