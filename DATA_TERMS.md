# Data Terms

This repository and its release assets are provided **for research purposes only** insofar
as they contain material derived from third-party research datasets. The repository's code
is Apache-2.0 (see `LICENSE`); that license applies to our code and does **not** re-license
any dataset or dataset-derived content. Dataset terms were verified against the official
sources on **2026-08-30**; URLs below record what was checked.

## What this repository does and does not redistribute

**Original dataset images are not redistributed**, with one exception: the five GUI example
photographs in `examples/photos/` (item 5 below). The release assets (`caches.tar.zst`,
`explainer.tar.zst`, `PanoCropping_demo.tar.zst`, `PolyHaven_panoramas_100.zip`) contain no
FCDB, FLMS or GAICD images. To run the benchmarks, obtain the images yourself from the
official sources listed below and honor each dataset's terms.

What the release assets and repository DO contain (full inventory:
`CACHE_MANIFEST.md`):

1. **Derived numerical feature caches** computed by frozen pretrained models over
   FCDB / FLMS / GAICD images and candidate crops: per-crop feature parquets (CLIP
   similarities, LAION and FGAesQ aesthetic scores, SAM encoder/decoder statistics),
   low-resolution derived maps (DINOv3 saliency maps, CLIP CAM heatmaps at 16x16,
   InSPyReNet subject masks at 256x256), and cached consultant-MLLM output boxes
   (coordinate JSONs). These are numerical measurements and model outputs, not image
   content; the original photographs cannot be reconstructed from them.
2. **Candidate and result boxes**: our own generated crop-candidate coordinates
   (DPP-K10 pools) and per-image predicted boxes with IoU/BDE scores
   (`inference/results/boxes_<cell>.csv`, `explanations_<cell>.json`) for the
   FCDB-test and FLMS test images, referenced by photo id.
3. **Dataset annotation files**: `FCDB-train.json` / `FCDB-test.json` (FCDB ground-truth
   crop annotations — released under MIT by the FCDB author, see below) and
   FLMS ground-truth annotations are not included; users obtain `flms_500_gt.mat` from
   the official source when running FLMS evaluation (see the FLMS note below).
4. **Poly Haven HDRIs**: the panorama demo takes CC0 HDRIs from https://polyhaven.com/hdris.
   The `.hdr` files are not redistributed: `run_pano.py --hdr auto` and the web app download them
   from Poly Haven on first use (`PanoCropping/polyhaven_hdr.json` pins each URL and MD5). The release asset `PolyHaven_panoramas_100.zip`
   ships the 100 user-study panoramas as 8192 x 4096 tone-mapped JPGs, and
   `PanoCropping_demo.tar.zst` ships downscaled previews and perspective renders of the same
   100 scenes; CC0 permits both. The eight GUI example panoramas in `examples/panoramas/` are
   files from the same set.
5. **GUI example photographs**: `examples/photos/` holds six test images, two from FCDB
   (`FCDB_<flickr id>.jpg`) and four from FLMS (`FLMS_<id>.jpg`), unmodified, so the GUI can be
   tried with one click. They are included for non-commercial research demonstration only;
   copyright remains with the original Flickr photographers, and the FCDB / FLMS terms below
   apply to them.

Users of this repository must: (a) obtain original images only from the official sources
below, (b) use the dataset-derived material here under the same research-use constraints
that govern the source datasets, and (c) not attempt to reconstruct, re-identify, or
redistribute original dataset content from the derived artifacts.

## Per-dataset terms

### FCDB — Flickr Cropping Dataset (Chen et al., WACV 2017)

- **Official source:** https://github.com/yiling-chen/flickr-cropping-dataset (verified
  2026-08-30). Images are downloaded from their original Flickr URLs via the repo's
  `download_images.py`; because many Flickr originals have expired, the author also shares
  a Google Drive mirror with the statement: "these images are intended for research
  purposes only."
- **Terms:** the repository (scripts + annotation JSONs) is MIT-licensed
  (Copyright (c) 2016, Yi-Ling Chen). The images themselves are Flickr photographs whose
  copyright remains with the original photographers; the FCDB repo grants no image
  redistribution rights beyond the research-only mirror.
- **What we redistribute:** the annotation splits (`FCDB-train.json`, `FCDB-test.json`,
  derived from the MIT-licensed annotations — the MIT copyright notice is retained via
  this file), derived per-crop feature caches, and per-image result boxes. No images.
- **Your obligations:** download the images via the official repo; treat them as
  research-use-only; credit the FCDB paper.

### FLMS (Fang et al., ACM MM 2014)

- **Official source:** the authors’ direct [annotation archive](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_gt/release_data.tar)
  and [image archive](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_image/image.tar).
  The project index is gone; the data objects remain served.
- **Dataset readme terms:** "This dataset is only for research purpose".
  Bounding-box order is (height_min, width_min, height_max, width_max);
  0 = first pixel; negative boxes must be ignored. Images are Flickr crawls
  provided by the authors on request.
- **What we redistribute:** derived per-crop feature caches and result boxes only. No images
  and no ground-truth annotation file. Obtain `flms_500_gt.mat` from the official FLMS source,
  place it at `caches/flms_500_gt.mat`, and see `tools/FLMS_GT.md`.
- **Your obligations:** obtain the 500 images from the authors or their official
  distribution; use everything FLMS-derived for research only; credit the FLMS paper.

### GAICD — Grid-Anchor-based Image Cropping Database (Zeng et al.)

- **Official source:** https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping
  (conference version) and
  https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch (journal version);
  the dataset itself is downloaded from the authors' Google Drive links in those READMEs
  (verified 2026-08-30).
- **Terms:** the conference repo has no LICENSE file (all rights reserved by default);
  the journal repo's MIT license covers its code. No written dataset license or
  research-only clause is published; the README notes a **PCT patent** on the method.
  Treat the dataset as research-use-only academic material.
- **What we redistribute:** derived per-crop 60-feature training rows with our own
  generated candidate-box coordinates and annotation-derived training targets
  (`caches/gaicd_deep60f/`). No images, and no copy of the GAICD annotation files
  themselves.
- **Your obligations:** download GAICD from the official links; research use only; note
  the patent statement if you are considering commercial use of grid-anchor-style
  methods; credit the GAIC papers.

### Poly Haven HDRIs (panorama demo input)

- **Official source:** https://polyhaven.com/hdris
- **Terms:** CC0 1.0 Universal for all Poly Haven assets
  (https://polyhaven.com/license, verified 2026-08-30): any purpose including commercial,
  redistribution allowed, no attribution required.
- **Obligations:** none. (Poly Haven's site content/logo are not CC0; only the assets.)

## Consultant-MLLM cached outputs

The cached consultant boxes (`venus_boxes_*.json`, `venus_multibox_*.json`,
`venus_promptens_*.json`) are decoded outputs — crop-box coordinates — produced by the
third-party Venus-Q-Stage2 checkpoint over FCDB/FLMS images. They are shipped as frozen
numerical inputs so the benchmark reproduces bit-exactly without any MLLM. The checkpoint
itself is not redistributed and carries no published license (see
`THIRD_PARTY_LICENSES.md`); the cached coordinates are used and shared here for research
verification only.

## Summary

- Research-use-only framing applies to everything in this repository that derives from
  FCDB, FLMS, or GAICD.
- No original images are redistributed; caches are derived numerical features, maps, and
  coordinates.
- Obtain the original datasets from the official sources linked above and honor each
  dataset's terms; those terms take precedence over this repository's Apache-2.0 code
  license for all dataset-derived material.


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
