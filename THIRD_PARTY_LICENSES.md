# Third-Party Licenses

This document inventories every third-party component the PhotoGuider repository depends on,
downloads at setup time, or ships artifacts derived from. The repository's own code is
licensed under Apache-2.0 (see `LICENSE`); the components below remain under their own
licenses or terms, which are not changed by inclusion here.

All licenses were verified against the official source on **2026-08-30**; the "Source /
verified at" column records the exact page checked. Type codes: **code** (software library),
**weights** (pretrained model parameters), **data** (dataset or dataset-derived artifact).
"What we do with it": *user-installed* (installed by the user via pip/venv, never
redistributed by us), *downloaded-at-setup* (fetched automatically by our scripts or on
first run, never redistributed by us), *vendored* (copies or derived artifacts shipped in
this repository or its release assets).

⚠ marks components whose terms are undeclared upstream or carry obligations beyond notice
retention. Dataset-derived artifacts we redistribute are additionally governed by
`DATA_TERMS.md`.

## Inventory

| Component | Type | License | Source / verified at (2026-08-30) | What we do with it |
|---|---|---|---|---|
| segment-anything (SAM, Meta) | code | Apache-2.0 | https://github.com/facebookresearch/segment-anything (main) | user-installed (`envs/sam/venv`) |
| SAM ViT-B checkpoint `sam_vit_b_01ec64.pth` | weights | Apache-2.0 (README: "The model is licensed under the Apache 2.0 license.") | https://github.com/facebookresearch/segment-anything (main) | downloaded-at-setup (user fetches into `envs/sam/weights/`) |
| transparent-background (pip) | code | MIT | https://github.com/plemeri/transparent-background (main) | user-installed |
| InSPyReNet (model behind transparent-background) | code | MIT | https://github.com/plemeri/InSPyReNet (main) | user-installed (via transparent-background) |
| ⚠ InSPyReNet checkpoints (downloaded by transparent-background) | weights | **Undeclared** (repo + model-zoo doc state no weight license; trained on academic SOD datasets, e.g. DUTS-TR) | https://github.com/plemeri/InSPyReNet/blob/main/docs/model_zoo.md (main) | downloaded-at-setup (auto-fetched on first `Remover()` use) |
| open_clip (mlfoundations) | code | MIT | https://github.com/mlfoundations/open_clip/blob/main/LICENSE (main) | user-installed (`envs/laion/venv`; loads `ViT-L-14, pretrained="openai"`) |
| OpenAI CLIP code (`clip` package, used in the `fgaesq` environment) | code | MIT | https://github.com/openai/CLIP (main) | user-installed |
| ⚠ CLIP ViT-L/14 weights (`openai/clip-vit-large-patch14`, also loaded by open_clip as `pretrained="openai"`) | weights | **No license tag on the HF model card**; card scopes intended use to research and calls any deployed use "out of scope" | https://huggingface.co/openai/clip-vit-large-patch14 (live card) | downloaded-at-setup (HF hub, first run) |
| LightGBM | code | MIT | https://github.com/microsoft/LightGBM (main) | user-installed |
| scikit-learn | code | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn (main) | user-installed |
| timm (pytorch-image-models) | code | Apache-2.0 (code; README warns pretrained weights carry per-model licenses) | https://github.com/huggingface/pytorch-image-models (main) | user-installed |
| transformers (Hugging Face) | code | Apache-2.0 | https://github.com/huggingface/transformers (main) | user-installed |
| PEFT (Hugging Face) | code | Apache-2.0 | https://github.com/huggingface/peft (main) | user-installed |
| bitsandbytes | code | MIT | https://github.com/bitsandbytes-foundation/bitsandbytes (main) | user-installed |
| Qwen2.5-VL-7B-Instruct (base VLM for the explanation module) | weights | Apache-2.0 (HF metadata license tag; no separate LICENSE file in the repo tree) | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (commit `cc594898137f460bfe9f0759e9844b3ce807cfb5`) | downloaded-at-setup (HF hub, first `explain_text.py` run); our LoRA adapter (`explainer`) is a derivative fine-tune |
| ⚠ Venus-Q-Stage2 (consultant MLLM, `popo28/Venus-Q-Stage2`) | weights | **None declared** — "No model card", no LICENSE/README in the file tree (commit `9d274cf`) | https://huggingface.co/popo28/Venus-Q-Stage2 | downloaded-at-setup (required for `--mode full`); we ship a 2-hunk patch to the upstream tokenizer file; the model weights are unchanged and not redistributed; its cached output boxes are vendored in `caches` |
| ⚠ DINOv3 ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m` via timm hub) | weights | **DINOv3 License** (custom Meta license; HF tag `dinov3-license`) — NOT MIT/Apache | https://github.com/facebookresearch/dinov3 + LICENSE.md; https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m (ungated mirror); gated original: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | downloaded-at-setup (timm pretrained hub, first live-saliency run); derived saliency caches are vendored |
| LAION aesthetic predictor linear head `sa_0_4_vit_l_14_linear.pth` | weights | MIT ("Copyright (c) 2022 LAION AI") | https://github.com/LAION-AI/aesthetic-predictor (main; LICENSE file present, file confirmed in repo) | downloaded-at-setup (user fetches into `envs/laion/weights/`) |
| ⚠ FGAesQ checkpoint + inference code (FG-IAA, CVPR 2026) | code + weights | **No LICENSE file visible** in the official repo; README says inference code + pretrained weights are "publicly available" (HF / Baidu links) without stated terms | https://github.com/yzc-ippl/FG-IAA (main) | user-installed / downloaded-at-setup (`envs/fgaesq/venv` + `envs/fgaesq/weights/FGAesQ.pt`); derived per-crop score caches are vendored |
| NIMA | — | *Not used.* The FGAesQ environment serves only FGAesQ; no NIMA weights ship or load anywhere in the release | — | — |
| FCDB (Flickr Cropping Dataset) | data | Annotations + scripts: MIT (c) 2016 Yi-Ling Chen; images: Flickr photos, author mirror "research purposes only" — see `DATA_TERMS.md` | https://github.com/yiling-chen/flickr-cropping-dataset (main) | images user-obtained; annotation JSONs + derived feature caches vendored |
| ⚠ FLMS (Fang et al., ACM MM'14) | data | Readme: "This dataset is only for research purpose"; direct data objects remain available — see `DATA_TERMS.md` | http://fangchen.org/proj_page/FLMS_mm14/ ; paper https://xiaohuishen.github.io/assets/cropping-mm2014.pdf | images and GT annotations user-obtained; only derived feature caches are vendored |
| ⚠ GAICD (Grid-Anchor-based Image Cropping Database) | data | **No dataset license** (conference repo has no LICENSE file; journal repo MIT covers code only; README notes a PCT patent on the method) — see `DATA_TERMS.md` | https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping (main); journal repo …-Pytorch (main) | images user-obtained; derived per-crop 60-feature rows vendored |
| Poly Haven HDRIs (panorama demo input) | data | CC0 1.0 Universal | https://polyhaven.com/license | `.hdr` files user-obtained; the 100 user-study panoramas (tone-mapped JPG) are the `PolyHaven_panoramas_100.zip` release asset, and downscaled previews and renders are vendored in the `PanoCropping_demo.tar.zst` release asset (`PanoCropping/README.md`) |
| Locked-environment PyPI stack (numpy, pandas, scipy, pyarrow, joblib, pillow, tqdm, opencv-python-headless, torch) | code | Permissive (BSD/MIT/Apache/HPND/PSF-family) per package | PyPI / respective upstream repos | user-installed |

## Per-component notes (obligations beyond notice-retention)

### Qwen2.5-VL-7B-Instruct — Apache-2.0 (clean for our LoRA adapter)
The 7B-Instruct card is tagged plain **apache-2.0** — it is *not* under the "Qwen LICENSE
AGREEMENT" / Qwen Research license used by some other Qwen2.5-VL sizes (3B, 72B), so there
are **no derivative-naming clauses** and no MAU thresholds. For our released
`explainer` LoRA adapter this requires: (a) include the Apache-2.0 license text with the
adapter distribution, (b) retain attribution (state the base model:
`Qwen/Qwen2.5-VL-7B-Instruct`; already recorded in `adapter_config.json`), (c) note that
the adapter is a fine-tuned derivative, (d) do not use Qwen/Alibaba trademarks to imply
endorsement. All four are satisfied by this document + the adapter config + the README.
The license is declared through the HF metadata tag; the model repository has no standalone
LICENSE file.

### Venus-Q-Stage2 and FGAesQ — non-commercial research

Both were released publicly with their papers and are used here for non-commercial
research: FGAesQ unmodified, Venus with a two-hunk tokenizer-compatibility patch.
No licence file is published upstream; users must check upstream terms before any
other use. Venus is required for `--mode full` (including `--venus-live` where exposed).
Weights are downloaded by users and are not redistributed here.

### ⚠ DINOv3 — custom Meta "DINOv3 License", not MIT/Apache
Verified in `facebookresearch/dinov3` LICENSE.md and the `dinov3-license` tag on the timm
hub mirror we actually load (`timm/vit_small_patch16_dinov3.lvd1689m`, ungated; the
official `facebook/dinov3-vits16-pretrain-lvd1689m` repo is gated). Key terms:
- Commercial use is allowed (royalty-free limited license).
- **Sticky derivative clause:** distributing DINO Materials *or derivative works of the
  weights* is only permitted under the DINOv3 License itself, with a copy of the agreement
  provided. We do not redistribute the weights or any fine-tune of them, so this clause is
  not triggered; if that ever changes, the derivative must ship under the DINOv3 License,
  not Apache-2.0.
- **Attribution:** research publications using DINO Materials must acknowledge the use.
- Use restrictions: no ITAR-subject activities, no military/weapons/espionage end uses,
  sanctions compliance; termination on breach or on filing infringement litigation
  against Meta.
- Outputs/derived features: the license text imposes no explicit redistribution
  restriction on model *outputs*; our vendored saliency caches are outputs (numerical
  maps/statistics), not derivative works of the weights.

### ⚠ CLIP ViT-L/14 weights — undeclared weight license + use-scope language
The GitHub code is MIT, but the HF card for `openai/clip-vit-large-patch14` carries **no
license tag**, describes the model as a research artifact, and states that "any deployed
use case of the model — whether commercial or not — is currently out of scope" (with
surveillance/facial recognition always out of scope). We do not redistribute the weights;
users download them from HF, and our repository only ships numerical similarity/CAM
features derived from them. Uses beyond research should follow the model card.

### ⚠ InSPyReNet checkpoints — MIT code, undeclared weight license
Both `transparent-background` and `InSPyReNet` are MIT, but the model-zoo documentation
attaches **no explicit license to the checkpoints**, which are trained on academic
salient-object-detection datasets (DUTS-TR etc., customarily research-use). We do not
redistribute the checkpoints (auto-downloaded by the pip package); we vendor derived
subject-mask caches. Commercial users should check the weight and training-data terms.

### FGAesQ (FG-IAA) — non-commercial research
Released publicly with its paper, used here unmodified for non-commercial research.
No licence file is published upstream; users must check upstream terms before any other
use. Users obtain its code and checkpoint directly; only derived score caches are vendored.

### segment-anything / SAM ViT-B — Apache-2.0 including the checkpoint
The repo README explicitly covers the model under Apache 2.0, so the `sam_vit_b_01ec64.pth`
checkpoint and features derived from it are unencumbered beyond Apache notice-retention.
Note the grant is via the README statement; there is no separate per-file weights license.

### timm — per-model weight licenses
timm's code is Apache-2.0, but its README warns that pretrained weights carry their own
licenses (some CC-BY-NC, some dataset-encumbered). The only timm-hub weights this project
loads are the DINOv3 ViT-S weights covered above; no other timm pretrained weights are
used.

### LAION aesthetic head — MIT
`sa_0_4_vit_l_14_linear.pth` is hosted in `LAION-AI/aesthetic-predictor`, which carries an
MIT LICENSE ("Copyright (c) 2022 LAION AI"). Retain the notice if the file is ever
redistributed; currently users download it themselves. Note the head is applied on top of
OpenAI CLIP ViT-L/14 embeddings, so the CLIP-weights note above applies to the combined
scorer.

### Apache-2.0 dependencies (transformers, PEFT, timm, segment-anything)
Standard Apache-2.0 obligations apply when redistributing them; we do not vendor any of
their source. None of the checked repos surfaced a NOTICE file requiring propagation. Our
own `NOTICE` file points here.

### Datasets (FCDB / FLMS / GAICD / Poly Haven)
See `DATA_TERMS.md` for the full terms, the exact list of what dataset-derived material
this repository redistributes, and the obligations on users. Flag summary:
- ⚠ **FLMS** is research-use only per its readme. We do not redistribute its annotation file;
  `tools/download_flms_gt.sh` fetches it from the authors' server (see `tools/FLMS_GT.md`).
- ⚠ **GAICD** publishes no dataset license (default all-rights-reserved) and notes a PCT
  patent on the method; we ship only derived feature rows (no images, no annotation
  files) for research use.
- FCDB annotations are MIT (notice retained in `DATA_TERMS.md`); images are user-obtained,
  research-use-only. Poly Haven HDRIs are CC0 — no obligations.


FCDB MIT notice (annotations and scripts):

The MIT License (MIT)

Copyright (c) 2016, Yi-Ling Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CLIP ViT-B/16 weights (OpenAI CLIP, loaded by FGAesQ) are downloaded on first use;
source: https://github.com/openai/CLIP. CLIP code is MIT; the model is released for research.
The validated ViT-B-16.pt digest is inventoried in `envs/asset_pins.json`.
