"""Cached model loaders shared by the per-crop extractors.

Each loader builds its model once and memoises it in ``_CACHE`` so a long-lived
worker process pays the (slow) construction cost only on the first job and reuses
the resident model for every subsequent image. Imports are done lazily inside each
loader because the four extractor environments do not all provide the same packages
(only the environment that actually calls a loader needs its imports).

Construction is byte-identical to the standalone extractors, so the numerics are
unchanged whether an extractor is run cold (one-shot) or warm (served by a worker).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_pins import revision
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # inference/
from env_config import asset as _asset

LAION_HEAD = _asset("laion_head", "envs/laion/weights/sa_0_4_vit_l_14_linear.pth")
FGAESQ_CKPT = _asset("fgaesq_checkpoint", "envs/fgaesq/weights/FGAesQ.pt")
FGAESQ_SOURCE = _asset("fgaesq_source", "envs/fgaesq/FGAesQ_Inference")
SAM_WEIGHTS = _asset("sam_checkpoint", "envs/sam/weights/sam_vit_b_01ec64.pth")

_CACHE = {}


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scaled_box(x1, y1, x2, y2, ref_W, ref_H, Wi, Hi):
    """Candidate box given on a (ref_W, ref_H) canvas -> integer pixel box on the Wi x Hi image
    actually opened, clamped to at least 1x1 inside it. Shared by every crop-scoring extractor."""
    ix1 = max(0, min(int(round(x1 / ref_W * Wi)), Wi - 1))
    iy1 = max(0, min(int(round(y1 / ref_H * Hi)), Hi - 1))
    ix2 = max(ix1 + 1, min(int(round(x2 / ref_W * Wi)), Wi))
    iy2 = max(iy1 + 1, min(int(round(y2 / ref_H * Hi)), Hi))
    return ix1, iy1, ix2, iy2


def get_clip():
    """(processor, model) for openai/clip-vit-large-patch14 (fp16). Used by ext_clip."""
    if "clip" not in _CACHE:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        dev = _device()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", revision=revision("clip"))
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", revision=revision("clip"), torch_dtype=torch.float16).to(dev).eval()
        _CACHE["clip"] = (proc, model)
    return _CACHE["clip"]


def get_laion():
    """(clip_model, preprocess, head, device) for the LAION aesthetic head.

    Used by both ext_laion and score_crops_laion so the ViT-L backbone is loaded once."""
    if "laion" not in _CACHE:
        import torch
        import open_clip
        dev = _device()
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_model = clip_model.eval().to(dev)
        head = torch.nn.Linear(768, 1)
        head.load_state_dict(torch.load(LAION_HEAD, map_location=dev, weights_only=True))
        head = head.eval().to(dev)
        _CACHE["laion"] = (clip_model, preprocess, head, dev)
    return _CACHE["laion"]


def _seed_all(s=42):
    import random
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_fgaesq():
    """(model, proc, device) for FGAesQ + DiffToken.

    Used by both ext_fgaesq and score_crops_fgaesq. The random-patch seeding
    discipline (seed before init and again before every batch) is preserved by the
    callers, so a resident model gives results identical to a fresh process."""
    if "fgaesq" not in _CACHE:
        import sys
        import torch
        sys.path.insert(0, str(FGAESQ_SOURCE))
        _seed_all()
        from utils.FGAesQ import FGAesQ
        from utils.DiffToken import DiffToken
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FGAesQ(pretrained_path=None).to(dev)
        model.load_state_dict(torch.load(FGAESQ_CKPT, map_location=dev), strict=False)
        model.eval()
        proc = DiffToken(clip_model=model.clip_model, patch_selection="random")
        _CACHE["fgaesq"] = (model, proc, dev)
    return _CACHE["fgaesq"]


def get_sam():
    """SAM ViT-B model on the active device. Shared by every SAM extractor
    (encoder features, decoder features, and segment-everything overlay) so the
    checkpoint is loaded once instead of once per stage."""
    if "sam" not in _CACHE:
        import torch
        from segment_anything import sam_model_registry
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHTS)).to(dev).eval()
        _memoise_encoder(sam.image_encoder)
        _CACHE["sam"] = sam
    return _CACHE["sam"]


def _memoise_encoder(enc, max_entries=160, max_bytes=3 << 29):
    """The image encoder sees the same full image several times per image (mean embedding, patch
    norms, the decoder's set_image, both extractor passes) and the same candidate batches in both
    passes. Its forward is memoised on the exact input tensor, so a repeat returns the stored output.
    The cache holds at most ``max_entries`` outputs and ``max_bytes`` of GPU memory, so a resident
    worker (the web app) does not grow across images; the limits fit one panorama scan (120 views
    plus their candidate batches) so the second extractor pass still hits."""
    import hashlib, threading
    forward, cache, lock = enc.forward, {}, threading.Lock()   # the decoder extractor calls in from several threads

    def cached(x):
        key = (tuple(x.shape), str(x.dtype), hashlib.sha1(x.detach().cpu().numpy().tobytes()).hexdigest())
        with lock:
            if key not in cache:
                out = forward(x)
                while cache and (len(cache) >= max_entries or
                                 sum(v.numel() * v.element_size() for v in cache.values()) + out.numel() * out.element_size() > max_bytes):
                    cache.pop(next(iter(cache)))
                cache[key] = out
            return cache[key]

    enc.forward = cached
