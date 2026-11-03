# PanoCropping — composition-aware view framing in 360° panoramas

Given an equirectangular panorama, choose where to point a virtual camera and what to frame there,
using the same photographic measurements as the 2D cropper. Coarse to fine:

- **Direction (coarse).** Render 120 overlapping perspective views (pans 0°..345° in 15° steps ×
  tilts −30°..30° in 15° steps, 40° horizontal FOV, 1024×1024) and score each with the additive
  view scorer. Interpolate a dense 1° score field, take its local maxima, suppress maxima within
  20° great-circle of a stronger one, keep the band within 15% of the range below the maximum, and
  pick at most three diverse hotspots by greedy DPP. The strongest hotspot is the shooting direction
  (for a user-study panorama under `--hdr auto`, the direction the study used).
- **Framing, refinement, consultation (fine).** Render a 1600×1067, 60° working view in that
  direction and hand it to `inference/photoguider.crop_image` (ranker, gated refinement, optional
  Venus consultant, assessments, optional explanation).
- **Camera.** Convert the chosen box into camera parameters: the box centre ray gives pan and tilt,
  the two horizontal edge rays at mid-height give the horizontal FOV, the pixel ratio gives the
  aspect. `final.png` is re-rendered natively from the panorama at a 1920-pixel long side: a
  resampling of the original 360° photo, no synthesis.

Everything is driven by composition alone; no scene labels, no manual view selection.

```
scripts/run_pano.py      the pipeline (CLI, see below)
scripts/polyhaven_hdr.py fetch and verify the Poly Haven HDR of a user-study panorama (standard library only)
polyhaven_hdr.json       per study scene: HDR URL, MD5 and size; SHA-256 of the 8-bit panorama
scripts/score_views.py   live saliency/mask maps + 60 deep features per view, and the scorer call
models/                  composition_scorer_60f.pkl + scorer.py (self-contained loader)
```

## Run

Everything runs from the `main` environment (extractor and explainer work is dispatched to the
configured environments automatically):

```bash
envs/main/venv/bin/python PanoCropping/scripts/run_pano.py panorama.jpg --out output/pano
envs/main/venv/bin/python PanoCropping/scripts/run_pano.py panorama.jpg --mode full --explain --out output/pano
envs/main/venv/bin/python PanoCropping/scripts/run_pano.py royal_esplanade.jpg --hdr auto --out output/pano   # study panorama
```

**Exposure from the source HDR.** The coarse scan scores the 8-bit panorama, where one exposure covers
every view. In the user study the working view and the final shot were then rendered from the
panorama's linear HDR and exposed for that view (median luminance to middle grey, AgX curve, the
final shot reusing the working view's gain). `--hdr panorama.hdr` does the same. `--hdr auto` does
it for the 100 user-study panoramas: the input is identified by its SHA-256 against
[`polyhaven_hdr.json`](polyhaven_hdr.json), and the matching 8k HDR is **downloaded from Poly Haven
on first use** (CC0, about 100 MB each, MD5 verified) into `~/.cache/photoguider/hdr` (set
`PHOTOGUIDER_HDR_DIR` to change it). Any other panorama is rendered from the 8-bit file itself. To
download ahead of time: `python3 PanoCropping/scripts/polyhaven_hdr.py <scene or panorama.jpg>`.
`result.json` records the choice in `panorama.exposure`.

**Direction of a study panorama.** The user study's coarse scans ran in an environment that no
longer exists. Its hotspot scores sit about one point below a scan today, and where the two
strongest hotspots are within a few tenths of a point a scan today can rank them the other way.
`--hdr auto` therefore also shoots a user-study panorama from the direction the study used
(`direction` in `polyhaven_hdr.json`). The scan still runs; the study direction is listed first in
`hotspots` with this run's score and the log reports where this run ranked it. `--scan-direction`
shoots from this run's strongest hotspot instead. `result.json` records the choice in
`panorama.direction`.

Outputs: `working_view.png`, `final.png`, `result.json`. The JSON follows
[`OUTPUT_SCHEMA.md`](../OUTPUT_SCHEMA.md): `box` is in working-view pixels, the assessments,
geometry and explanation describe the crop inside the working view, and the `panorama` block holds
the final camera (`pan` signed, 0 = panorama centre column; `tilt`; horizontal `fov`; render
`width`/`height`) and the ordered `hotspots` with their raw view scores (the first is the shooting
direction).

## The view scorer (`models/`)

A token-grouped additive model (40,504 parameters) over the **same 60 features the 2D cropper's
ranker uses**: per-feature contributions are grouped into the four tokens (subject / composition /
aesthetics / integrity), summed to a scalar composition score and mapped by isotonic regression to
the GAICD 1–5 mean-opinion scale. It is trained on GAICD crop rankings (pairwise loss) and reaches
SRCC 0.763 on the GAICD test split; the artifact metadata records the validation-split value
(0.742). The loader is self-contained: `from scorer import load, build_scorer, score_60f`.

In the coarse scan each view's "crop box" is the whole frame, so the box-geometry features are
constant across the views of one panorama and only shift the score uniformly; cross-view differences
come from the content features (saliency, colour, CLIP, LAION, FGAesQ, SAM). Hotspot selection
uses the raw score.

## Geometry conventions

Camera axes x right, y up, z forward, zero roll, rotation `Ry(pan) @ Rx(-tilt)`; pan 0 faces the
panorama centre column and positive pan moves toward increasing columns; positive tilt looks up.
Focal length `(width/2)/tan(fov/2)`, principal point `((width-1)/2, (height-1)/2)`. The
constants live in `inference/_lib/pano.py`; `tests/test_pano_explain.py` checks hotspot selection on
synthetic fields and the box → camera conversion.

## Notes

- Scanning 120 views costs roughly 400–510 s per panorama on one GPU; the fine stage about 60 s
  (fast mode).
- Demo panoramas are CC0 from [Poly Haven](https://polyhaven.com/hdris). The 100 user-study
  panoramas themselves (8192 x 4096 tone-mapped JPGs, 900 MB) are the release asset
  `PolyHaven_panoramas_100.zip`:
  `gh release download release -R NTUneillee/PhotoGuider -p PolyHaven_panoramas_100.zip`
  (SHA-256 `3f71f9c9669c3ffaed2f54c8cd00f4467e36a08a8256b9234a1bb87eaa39e2f0`). The results on the 100
  user-study panoramas (preview, working view, final shot and `result.json` per scene) ship as the
  release asset `PanoCropping_demo.tar.zst` rather than in this repository:
  `gh release download release -R NTUneillee/PhotoGuider -p PanoCropping_demo.tar.zst`
  (SHA-256 `a1f1e34da0dd6089119585ee8928f410f5ac0fd35fd0a74e8a0e08de9588f9c8`).
