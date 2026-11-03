"""Live DINOv3 saliency — byte-exact reproduction of the frozen V_MIX saliency cache.

The cached V_MIX map (`saliency_maps_dv3s.pkl`) is
  vmix = 0.5*minmax(V1) + 0.5*minmax(PC1)
  V1  = the DINOv3 attention variant (dinov3_saliency.compute_variants["V1"])
  PC1 = |first per-image patch-PCA component| (dinov3_saliency.pca_per_image)
NB: the *variants* V1 extractor must be used (not the historical V1_orig map, which differs at
corr ~0.90). Verified live vmix == cache at corr 1.00000. Mask = InSPyReNet 256x256."""
import math
import numpy as np, torch
from PIL import Image
import dinov3_saliency as DV3   # vendored V1 attention variant + patch PCA
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_STATE = {}


def _model():
    if "m" not in _STATE:
        model, n_prefix = DV3.load_model()
        _STATE["m"] = (model, n_prefix, DV3.get_transform())
    return _STATE["m"]


def _nm(x):  # combine_v1_pc1.normalize_minmax
    x = x.astype(np.float32); lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _grid(P):
    h = int(round(math.sqrt(P))); w = P // h
    if h * w != P:
        for hp in range(int(math.sqrt(P)) + 5, 0, -1):
            if P % hp == 0: return hp, P // hp
    return h, w


@torch.no_grad()
def build_saliency(image_path):
    model, n_prefix, tf = _model()
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(_DEV)
    # V1 via variants extractor (the one that built V_MIX)
    captured, originals, target = DV3.install_attention_hooks(model, DV3.N_LAYERS_LAST)
    try:
        captured.clear(); _ = model(x)
        v1 = DV3.compute_variants(captured, n_prefix, target)["V1"][0]      # (P,)
    finally:
        DV3.restore_hooks(model, originals)
    # PC1 via pca extractor
    patches = DV3.extract_patches(model, x, n_prefix)                        # (1,P,D)
    pcs = DV3.pca_per_image(patches[0], n_components=3).cpu().numpy()
    pc1 = np.abs(pcs[:, 0])
    h, w = _grid(v1.shape[0])
    v1_2d = v1.reshape(h, w).astype(np.float32)
    pc1_2d = pc1.reshape(h, w).astype(np.float32)
    vmix_2d = (0.5 * _nm(v1_2d) + 0.5 * _nm(pc1_2d)).astype(np.float32)
    return vmix_2d, v1_2d


def build_mask(image_path):
    from transparent_background import Remover
    if "rem" not in _STATE: _STATE["rem"] = Remover()
    img = Image.open(image_path).convert("RGB")
    m = _STATE["rem"].process(img, type="map")        # PIL
    m = np.asarray(m.convert("L").resize((256, 256))).astype(np.uint8)
    return m
