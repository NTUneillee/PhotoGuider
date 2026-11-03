"""Live CLIP-ViT-L/14 refine signals for one image:
  - CAM heatmap (16x16x9 over composition prompts) -> pan cam_center
  - composition-type vector ct (19-d: WIDE6+TIGHT4+KUPCP9 image sims)
Both per-image (no group dependency) -> fully reproducible for in-the-wild.
Single CLIP-L load. Uses the configured main environment (transformers)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_pins import revision
import torch, numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_S = {}

# CAM prompts (9) — heat order = PK (identical to the cached benchmark heatmaps)
CAM_PROMPTS = [
    "a professional photograph with well-balanced composition", "a well-balanced photograph",
    "a photograph with strong leading lines", "a photograph following the rule of thirds",
    "a symmetric photograph", "a photograph with effective negative space",
    "a centered photograph", "a dynamic and visually striking photograph",
    "a highly aesthetic photograph"]
# composition-type prompts (WIDE6 + TIGHT4 + KUPCP9) — ct order (identical to the cached vectors)
WIDE = ["a wide landscape photograph with open space and background",
        "a scenic photo with leading lines receding into the distance",
        "a horizontal wide-angle composition with a broad field of view",
        "a photo with strong diagonal lines and depth",
        "a photo of a repeating pattern filling the whole scene",
        "an environmental wide shot showing the subject within its surroundings"]
TIGHT = ["a tight close-up of a single subject filling the frame",
         "a centered portrait of one dominant subject",
         "a minimal photo with a single subject and very little background",
         "a close cropped shot focusing on one object"]
KUPCP = ["rule of thirds composition", "centered composition", "horizontal composition",
         "symmetric composition", "diagonal composition", "curved composition",
         "vertical composition", "triangle composition", "pattern composition"]
CT_PROMPTS = WIDE + TIGHT + KUPCP


def _load():
    if "clip" not in _S:
        dt = torch.float16 if _DEV == "cuda" else torch.float32
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", revision=revision("clip"), torch_dtype=dt).to(_DEV).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", revision=revision("clip"))
        # precompute normalized text features for CAM (projected) and CT (text_projection)
        with torch.no_grad():
            ti = proc(text=CAM_PROMPTS, return_tensors="pt", padding=True).to(_DEV)
            te = clip.get_text_features(**ti)
            if hasattr(te, "pooler_output"):
                te = te.pooler_output
                if hasattr(clip, "text_projection"): te = clip.text_projection(te)
            te = te / (te.norm(dim=-1, keepdim=True) + 1e-8)
            tj = proc(text=CT_PROMPTS, return_tensors="pt", padding=True).to(_DEV)
            to = clip.text_model(input_ids=tj["input_ids"], attention_mask=tj.get("attention_mask"))
            tf = clip.text_projection(to.pooler_output); tf = tf / (tf.norm(dim=-1, keepdim=True) + 1e-8)
        _S["clip"] = clip; _S["proc"] = proc; _S["cam_te"] = te.float().cpu().numpy(); _S["ct_tf"] = tf
    return _S["clip"], _S["proc"]


@torch.no_grad()
def cam_and_ct(image_path):
    """Return (heat 16x16x9 float32, ct 19-d float32) for one image."""
    clip, proc = _load()
    img = Image.open(image_path).convert("RGB")
    inp = proc(images=[img], return_tensors="pt").to(_DEV)
    if _DEV == "cuda": inp["pixel_values"] = inp["pixel_values"].half()
    vo = clip.vision_model(pixel_values=inp["pixel_values"])
    # CAM: patch tokens -> visual_projection -> sims to CAM text feats -> 16x16x9
    pats = vo.last_hidden_state[:, 1:, :]
    proj = pats @ clip.visual_projection.weight.T; proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
    pn = proj.float().cpu().numpy()[0]                          # (256,768)
    sims = pn @ _S["cam_te"].T                                  # (256,9)
    heat = sims.reshape(16, 16, len(CAM_PROMPTS)).astype(np.float32)
    # CT: pooled image feat -> sims to CT text feats (19,)
    imf = clip.visual_projection(vo.pooler_output); imf = imf / (imf.norm(dim=-1, keepdim=True) + 1e-8)
    ct = (imf @ _S["ct_tf"].T).float().cpu().numpy()[0].astype(np.float32)
    return heat, ct
