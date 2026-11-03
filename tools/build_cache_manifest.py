#!/usr/bin/env python3
"""Regenerate the public cache inventory."""
import argparse
import hashlib
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


DESCRIPTIONS = {
    'FCDB-test.json': 'FCDB test split (345 images) with GT crops',
    'FCDB-train.json': 'FCDB training split (1,369 images) with GT crops',
    'aes_delta_ens.parquet': 'per-image aesthetic-delta LUT over expansion scales (aes gate driver)',
    'aes_gold_delta.parquet': 'per-image golden-ratio-delta LUT over expansion scales (gold gate driver)',
    'aesthetic_integrity_features.parquet': 'FCDB per-crop SAM decoder features (axis 5, thirds prompts)',
    'aspect_free_cache/aspect_free_clipmp_flms.parquet': 'aspect-free per-crop CLIP-L multi-prompt similarities',
    'aspect_free_cache/aspect_free_consultant_blend_candidates_flms_core_features.parquet': 'aspect-free twin-candidate manifest with core features',
    'aspect_free_cache/aspect_free_consultant_blend_candidates_flms_core_features_features_only.parquet': 'aspect-free twin features (manifest companion)',
    'aspect_free_cache/aspect_free_consultant_blend_clipmp_flms.parquet': 'aspect-free twin CLIP-L similarities',
    'aspect_free_cache/aspect_free_consultant_blend_fgaesq_flms.parquet': 'aspect-free twin FGAesQ scores',
    'aspect_free_cache/aspect_free_consultant_blend_laion_flms.parquet': 'aspect-free twin LAION scores',
    'aspect_free_cache/aspect_free_consultant_blend_segment_decoder_flms.parquet': 'aspect-free twin SAM decoder features',
    'aspect_free_cache/aspect_free_consultant_blend_segment_encoder_flms.parquet': 'aspect-free twin SAM encoder features',
    'aspect_free_cache/aspect_free_fgaesq_flms.parquet': 'aspect-free per-crop FGAesQ scores',
    'aspect_free_cache/aspect_free_flms_pool.parquet': 'aspect-free FLMS candidate pool (fair free-form protocol)',
    'aspect_free_cache/aspect_free_flms_selected_candidates.parquet': 'aspect-free FLMS DPP-K10 candidates',
    'aspect_free_cache/aspect_free_laion_flms.parquet': 'aspect-free per-crop LAION scores',
    'aspect_free_cache/aspect_free_segment_decoder_flms.parquet': 'aspect-free per-crop SAM decoder features',
    'aspect_free_cache/aspect_free_segment_encoder_flms.parquet': 'aspect-free per-crop SAM encoder features',
    'aspect_free_cache/aspect_free_selection_surface.csv': 'free-form selection surface',
    'aspect_free_cache/dpp_q_lgbm.pkl': 'canonical DPP quality model (LightGBM, frozen)',
    'clip_composition_type.pkl': 'per-image CLIP composition-type vectors (wide/tight/composition prompts)',
    'clip_multi_prompt_features.parquet': 'FCDB per-crop CLIP-L multi-prompt similarities',
    'consultant_blend_candidates_fcdb_core_features.parquet': 'FCDB twin-candidate manifest with core features',
    'consultant_blend_candidates_flms_core_features.parquet': 'FLMS twin-candidate manifest with core features',
    'consultant_blend_clipmp_fcdb.parquet': 'twin per-crop CLIP-L multi-prompt similarities',
    'consultant_blend_clipmp_flms.parquet': 'twin per-crop CLIP-L multi-prompt similarities',
    'consultant_blend_fgaesq_fcdb.parquet': 'twin per-crop FGAesQ scores',
    'consultant_blend_fgaesq_flms.parquet': 'twin per-crop FGAesQ scores',
    'consultant_blend_laion_fcdb.parquet': 'twin per-crop LAION scores',
    'consultant_blend_laion_flms.parquet': 'twin per-crop LAION scores',
    'consultant_blend_segment_decoder_fcdb.parquet': 'twin per-crop SAM decoder features',
    'consultant_blend_segment_decoder_flms.parquet': 'twin per-crop SAM decoder features',
    'consultant_blend_segment_encoder_fcdb.parquet': 'twin per-crop SAM encoder features',
    'consultant_blend_segment_encoder_flms.parquet': 'twin per-crop SAM encoder features',
    'fcdb_attention_saliency.pkl': 'FCDB DINOv3 V1 saliency maps (historical V1 variant; single-image cached mode)',
    'fcdb_attention_saliency_variants.pkl': 'FCDB DINOv3 V1 saliency maps (variants extractor; twin featurization)',
    'fcdb_candidate_photographic_features.parquet': 'photographic per-crop features for the full candidate pool',
    'fcdb_clip_cam_heatmaps.npz': 'FCDB per-image CLIP-L CAM heatmaps (16x16x9)',
    'fcdb_conditioned_candidate_pool.parquet': 'FCDB full conditioned candidate pool (single-image cached-pool mode)',
    'fcdb_laion_aesthetic_scores.parquet': 'FCDB per-crop LAION aesthetic scores',
    'fcdb_selected_candidates.parquet': 'FCDB canonical DPP-K10 candidate boxes (box<->rank ground truth)',
    'fgaesq_features.parquet': 'FCDB per-crop fine-grained aesthetic (FGAesQ) scores',
    'flms_aesthetic_integrity_features.parquet': 'FLMS per-crop SAM decoder features (axis 5, thirds prompts)',
    'flms_attention_saliency.pkl': 'FLMS DINOv3 V1 saliency maps',
    'flms_clip_cam_heatmaps.npz': 'FLMS per-image CLIP-L CAM heatmaps',
    'flms_clip_multi_prompt_features.parquet': 'FLMS per-crop CLIP-L multi-prompt similarities',
    'flms_fgaesq_scores.parquet': 'FLMS per-crop FGAesQ scores',
    'flms_handcrafted_geometry_features.parquet': 'FLMS per-crop SAM encoder features (hand-crafted scalars)',
    'flms_inspy_masks.pkl': 'FLMS InSPyReNet subject masks (256x256)',
    'flms_laion_scores.parquet': 'FLMS per-crop LAION aesthetic scores',
    'flms_saliency_combined_sum.pkl': 'FLMS DINOv3 V_MIX saliency maps',
    'flms_selected_candidates.parquet': 'FLMS canonical DPP-K10 candidate boxes (conditioned reference)',
    'flms_spatial_composition_features.parquet': 'FLMS per-crop SAM encoder features (spatial patch norms)',
    'flms_subject_composition_features.parquet': 'FLMS per-crop SAM decoder features (axis 3, auto-mask object stats)',
    'flms_subject_composition_integrity_features.parquet': 'FLMS per-crop SAM decoder features (axes 1+2+4)',
    'flms_subject_mask_features.parquet': 'FLMS per-crop SAM encoder features (point-prompt mask stats)',
    'gaicd_selected_candidates.parquet': 'GAICD mixed-training candidate pool (box<->rank ground truth for the self-check)',
    'gaicd_subset_order.parquet': 'seed-42 shuffle order of GAICD-train images (subset_rank < 1000 = the mixed rows)',
    'handcrafted_geometry_features.parquet': 'FCDB per-crop SAM encoder features (hand-crafted scalars)',
    'inspy_masks_fcdb.pkl': 'FCDB InSPyReNet subject masks (256x256)',
    'saliency_maps_dv3s.pkl': 'FCDB DINOv3 V_MIX saliency maps (single-image cached mode)',
    'spatial_composition_features.parquet': 'FCDB per-crop SAM encoder features (spatial patch norms)',
    'subject_composition_features.parquet': 'FCDB per-crop SAM decoder features (axis 3, auto-mask object stats)',
    'subject_composition_integrity_features.parquet': 'FCDB per-crop SAM decoder features (axes 1+2+4)',
    'subject_mask_features.parquet': 'FCDB per-crop SAM encoder features (point-prompt mask stats)',
    'venus_boxes_fcdb.json': 'consultant greedy boxes, FCDB test',
    'venus_boxes_flms.json': 'consultant greedy boxes, FLMS',
    'venus_multibox_fcdb_te.json': 'consultant temperature-sample boxes (5 slots), FCDB test',
    'venus_multibox_flms.json': 'consultant temperature-sample boxes (5 slots), FLMS',
    'venus_promptens_fcdb_te.json': 'consultant prompt-ensemble boxes (4 paraphrases), FCDB test',
    'venus_promptens_flms.json': 'consultant prompt-ensemble boxes (4 paraphrases), FLMS',
}


def purpose(rel):
    name = rel.as_posix()
    if name.startswith("gaicd_deep60f/deep60f_"):
        return 'GAICD per-crop 60-feature rows (features and box share one row)'
    try:
        return DESCRIPTIONS[name]
    except KeyError as exc:
        raise KeyError(f"no public purpose recorded for {name}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=str(Path(__file__).resolve().parents[1] / "caches"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "CACHE_MANIFEST.md"))
    a = ap.parse_args()
    root, out = Path(a.caches), Path(a.out)
    excluded = {".extracted.sha256", "flms_500_gt.mat"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded)
    lines = ["# Final cache archive inventory", "",
             "The `caches.tar.zst` asset in the GitHub release tagged `release` extracts to `caches/`.",
             "It excludes FLMS ground truth; see `tools/FLMS_GT.md`.",
             "Regenerated with `tools/build_cache_manifest.py`.", "",
             "| file | bytes | SHA-256 | purpose |", "|---|---:|---|---|"]
    for path in files:
        rel = path.relative_to(root)
        lines.append(f"| `{rel.as_posix()}` | {path.stat().st_size} | `{digest(path)}` | {purpose(rel)} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
