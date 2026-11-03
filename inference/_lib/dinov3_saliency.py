"""DINOv3 saliency primitives (attention variants + per-image patch PCA).

Model loader + attention hooks + the V1 attention-variant computation + patch PCA used by
_lib/feat_saliency.py to rebuild the V_MIX saliency live for in-the-wild images. Byte-identical
extracts of the validated benchmark extractors; the DINOv3 weights come from timm's pretrained
hub (external model asset)."""
import time
import torch

DINO_MODEL = "vit_small_patch16_dinov3.lvd1689m"
INPUT_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LAYERS_LAST = 4


def load_model():
    import timm
    print(f"[DINOv3] loading {DINO_MODEL}", flush=True)
    t0 = time.time()
    model = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0)
    model.eval().to(DEVICE)
    n_prefix = getattr(model, "num_prefix_tokens", 1 + 4)
    print(f"  loaded in {time.time()-t0:.1f}s  prefix_tokens={n_prefix}  "
          f"n_layers={len(model.blocks)}  device={DEVICE}", flush=True)
    return model, n_prefix


def get_transform():
    import timm
    return timm.data.create_transform(
        input_size=(3, INPUT_SIZE, INPUT_SIZE),
        crop_pct=1.0,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bicubic",
        is_training=False,
    )


def install_attention_hooks(model, n_last):
    """Patch last n_last EvaAttention modules to capture softmax attention.

    Returns a dict captured[layer_idx] = (B, H, T, T) per call.
    Caller must clear `captured` between forward passes.
    """
    captured = {}
    n_layers = len(model.blocks)
    target_layers = list(range(n_layers - n_last, n_layers))
    originals = {}

    def make_patch(layer_idx, attn_mod):
        orig = attn_mod.forward
        originals[layer_idx] = orig

        def patched(x_in, *a, **k):
            B, N, C = x_in.shape
            qkv = attn_mod.qkv(x_in)
            H = attn_mod.num_heads
            D = C // H
            qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
            q, k_, v = qkv[0], qkv[1], qkv[2]
            if hasattr(attn_mod, "q_norm") and attn_mod.q_norm is not None:
                q = attn_mod.q_norm(q)
            if hasattr(attn_mod, "k_norm") and attn_mod.k_norm is not None:
                k_ = attn_mod.k_norm(k_)
            scale = D ** -0.5
            attn = (q @ k_.transpose(-2, -1)) * scale
            attn_soft = attn.softmax(dim=-1)
            captured[layer_idx] = attn_soft.detach()
            if hasattr(attn_mod, "attn_drop"):
                x_attn = attn_mod.attn_drop(attn_soft) @ v
            else:
                x_attn = attn_soft @ v
            x_out = x_attn.transpose(1, 2).reshape(B, N, C)
            if hasattr(attn_mod, "proj"): x_out = attn_mod.proj(x_out)
            if hasattr(attn_mod, "proj_drop"): x_out = attn_mod.proj_drop(x_out)
            return x_out
        return patched

    for li in target_layers:
        model.blocks[li].attn.forward = make_patch(li, model.blocks[li].attn)
    return captured, originals, target_layers


def restore_hooks(model, originals):
    for li, orig in originals.items():
        model.blocks[li].attn.forward = orig


def compute_variants(captured, n_prefix, target_layers):
    """The V1 saliency variant (head-mean CLS-to-patch attention of the last hooked layer).

    Returns {"V1": (B, P) numpy array}. The benchmark extractor also produced four other
    variants from the same hooks; only V1 feeds V_MIX, so only V1 is computed here.
    """
    last_li = target_layers[-1]
    last_attn = captured[last_li]                                # (B, H, T, T)
    cls_to_patch_last = last_attn[:, :, 0, n_prefix:]            # (B, H, P)

    # V1: head-mean × layer-last
    v1 = cls_to_patch_last.mean(dim=1)                            # (B, P)

    return {"V1": v1.cpu().numpy()}


def extract_patches(model, x, n_prefix):
    """Returns patch tokens (B, P, D) — drop CLS + register prefix."""
    feats = model.forward_features(x)   # (B, T, D)
    return feats[:, n_prefix:, :]


def pca_per_image(patches: torch.Tensor, n_components: int = 3):
    """Per-image PCA via SVD. Input (P, D). Returns (P, n_components) projections.

    No centering bias — subtract per-image mean first.
    """
    P, D = patches.shape
    mean = patches.mean(dim=0, keepdim=True)
    X = patches - mean
    # SVD: X = U S V^T,  PCs = X @ V[:, :k]
    # Use torch.linalg.svd
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except Exception:
        # Fallback CPU
        U, S, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        U = U.to(patches.device); S = S.to(patches.device); Vh = Vh.to(patches.device)
    # PCs = U[:, :k] * S[:k] (i.e. X projected onto top-k eigenvectors)
    pcs = U[:, :n_components] * S[:n_components].unsqueeze(0)
    return pcs
