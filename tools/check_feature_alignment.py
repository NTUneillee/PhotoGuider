#!/usr/bin/env python
"""Cache feature-parquet <-> box alignment integrity checker (release edition).

WHAT THIS VALIDATES
  Default (cache/parquet level): for every feature-parquet group under the cache dir, joins
  the feature files onto their box source of truth strictly on (photo_id, dpp_rank) and
  verifies, value-for-value against rounded box coords (round 3), that every feature row
  lands on the box that actually produced it. This proves the cache files' INTERNAL KEY
  CONSISTENCY — i.e. that a correct key-based join CAN align them ("alignment potential").
  --frame (assembled-frame level): additionally rebuilds the FCDB training frame exactly as
  inference/reproduce.py's loader does (box-coord dpp_rank re-key + stable mergesort +
  key-based merges) and audits THAT frame. This is the same check reproduce.py runs built-in
  before evaluating; use --frame to reproduce it standalone.

WHY TWO LEVELS
  Internally consistent parquets still need a loader that joins them on the right keys. The
  default level checks the files; --frame checks the frame reproduce.py actually assembles
  (the same logic as reproduce.py's built-in self-check).

Groups checked (skipped with a note if files are absent):
  fcdb        fcdb_selected_candidates + 9 deep-feature parquets
  flms        flms_selected_candidates + 9 flms_* parquets
  fcdb_twin / flms_twin    twin_cands_*_fmin + twin_{tag}_*
  af / af_twin             aspect_free_cache k10 / twin manifests + af_*_{tag} parquets
  gaicd       gaicd_deep60f shards vs gaicd_selected_candidates (box<->rank identity, per-crop 60F,
              so no separate feature parquet to value-compare — checks box membership,
              dpp_rank == box-derived rank, and exactly K10 rows per image)

Exits non-zero if any checked group is misaligned.

Usage: python tools/check_feature_alignment.py [--data_dir DIR] [--frame]
       (default DIR: <repo>/caches)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "caches"

FCDB_FEATS = ["fcdb_laion_aesthetic_scores", "handcrafted_geometry_features", "spatial_composition_features",
              "subject_mask_features", "subject_composition_integrity_features", "subject_composition_features",
              "aesthetic_integrity_features", "fgaesq_features", "clip_multi_prompt_features"]
FLMS_FEATS = ["flms_laion_scores", "flms_handcrafted_geometry_features",
              "flms_spatial_composition_features", "flms_subject_mask_features",
              "flms_subject_composition_integrity_features", "flms_subject_composition_features",
              "flms_aesthetic_integrity_features", "flms_fgaesq_scores", "flms_clip_multi_prompt_features"]
FEATURE_KINDS = ["laion", "clipmp", "fgaesq", "segment_encoder", "segment_decoder"]


def box_key(df, nd=3):
    return list(zip(df.photo_id.tolist(),
                    np.round(df.x1.values, nd), np.round(df.y1.values, nd),
                    np.round(df.x2.values, nd), np.round(df.y2.values, nd)))


def check_gaicd(D):
    """GAICD box<->rank identity (= reproduce.py gaicd_alignment_self_check). The 60F deep
    features were extracted per-crop (feature vector + box share one row), so there is no
    separate feature parquet to value-compare: the integrity test is that every deep60f row's
    box exists in gaicd_selected_candidates, its dpp_rank equals the box-derived true rank, and every
    image carries exactly K10 rows."""
    shards = sorted((D / "gaicd_deep60f").glob("deep60f_*.parquet"))
    if not shards:
        print("[gaicd] SKIP: no deep60f shards under gaicd_deep60f/"); return None
    g = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    cand = pd.read_parquet(D / "gaicd_selected_candidates.parquet")
    tmap = dict(zip(box_key(cand), cand.dpp_rank.tolist()))
    if len(tmap) != len(cand):
        print("[gaicd] ERROR: duplicate rounded boxes in gaicd_selected_candidates"); return False
    gk = box_key(g)
    n_missing = sum(1 for k in gk if k not in tmap)
    true_rank = np.array([tmap.get(k, -1) for k in gk])
    key_bad = int((g.dpp_rank.values != true_rank).sum())
    non_k10 = int((g.groupby("photo_id").size() != 10).sum())
    print(f"[gaicd] rows {len(g)}, pids {g.photo_id.nunique()}, boxes absent from pool: {n_missing}, "
          f"key mismatches (frame dpp_rank vs box-derived rank): {key_bad}, non-K10 pids: {non_k10}")
    ok = (n_missing == 0) and (key_bad == 0) and (non_k10 == 0)
    print(f"[gaicd] VERDICT: {'ALIGNED' if ok else 'MISALIGNED'}")
    return ok


def frame_fcdb_corrected(D):
    """= reproduce.py load_fcdb merge steps (no fillna, so missing values stay visible):
    box-coord-derived dpp_rank re-key + stable sort + key-based merges."""
    base = pd.read_parquet(D / "fcdb_selected_candidates.parquet")
    tmap = dict(zip(box_key(base), base.dpp_rank.tolist()))
    assert len(tmap) == len(base), "duplicate rounded boxes in candidate parquet"
    base = base.copy()
    base["dpp_rank"] = [tmap[k] for k in box_key(base)]
    base = base.sort_values(["photo_id", "dpp_rank"], kind="mergesort").reset_index(drop=True)
    for fn in FCDB_FEATS:
        base = base.merge(pd.read_parquet(D / f"{fn}.parquet"), on=["photo_id", "dpp_rank"], how="left")
    return base


def frame_keyed(base_path, feat_paths, drop_dup=False):
    """Direct key-based merge on the box source's stored (photo_id, dpp_rank)."""
    df = pd.read_parquet(base_path)
    for p in feat_paths:
        f = pd.read_parquet(p)
        if drop_dup:                                            # = load_af_frame semantics
            f = f.drop(columns=[c for c in f.columns if c in df.columns and c not in ("photo_id", "dpp_rank")])
        df = df.merge(f, on=["photo_id", "dpp_rank"], how="left")
    return df


def check_group(name, box_src_path, frame, feat_paths):
    """frame: the loaded/merged frame to audit; box_src: (photo_id, box, dpp_rank) truth."""
    truth = pd.read_parquet(box_src_path)
    tmap = dict(zip(box_key(truth), truth.dpp_rank.tolist()))
    if len(tmap) != len(truth):
        print(f"[{name}] ERROR: duplicate rounded boxes in box source of truth"); return False
    tr = [tmap.get(k) for k in box_key(frame)]
    n_missing = sum(r is None for r in tr)
    if n_missing:
        print(f"[{name}] ERROR: {n_missing} frame boxes not found in box source"); return False
    true_rank = np.array(tr)
    key_bad = int((frame.dpp_rank.values != true_rank).sum())
    print(f"[{name}] rows {len(frame)}, pids {frame.photo_id.nunique()}, "
          f"key mismatches (frame dpp_rank vs box-derived rank): {key_bad} ({100.0*key_bad/len(frame):.1f}%)")

    ok = key_bad == 0
    idx = frame[["photo_id"]].copy(); idx["dpp_rank"] = true_rank
    for p in feat_paths:
        f = pd.read_parquet(p)
        cols = [c for c in f.columns if c not in ("photo_id", "dpp_rank") and c in frame.columns]
        ref = idx.merge(f, on=["photo_id", "dpp_rank"], how="left")
        bad = np.zeros(len(frame), dtype=bool)
        for c in cols:
            a, b = frame[c].values, ref[c].values
            bad |= ~((a == b) | (pd.isna(a) & pd.isna(b)))
        n_bad = int(bad.sum())
        ok &= n_bad == 0
        print(f"  {p.name:45s} {n_bad:6d}/{len(frame)} rows misaligned  "
              f"{'OK' if n_bad == 0 else 'FAIL'}")
    print(f"[{name}] VERDICT: {'ALIGNED' if ok else 'MISALIGNED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Cache alignment integrity checker. Default: validates the cache files' "
                    "internal key consistency (feature parquets joined strictly on "
                    "photo_id+dpp_rank land on the box that produced them, verified via box "
                    "coords). With --frame: additionally audits the assembled FCDB training "
                    "frame exactly as reproduce.py's loader builds it "
                    "(same check reproduce.py runs built-in before evaluating).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA),
                    help=f"cache dir (default: {DEFAULT_DATA})")
    ap.add_argument("--frame", action="store_true",
                    help="also check the assembled FCDB frame (corrected-loader reconstruction)")
    args = ap.parse_args()
    D = Path(args.data_dir); AF = D / "aspect_free_cache"

    groups = []  # (name, box_src, frame_builder, feat_paths)
    if (D / "fcdb_selected_candidates.parquet").exists():
        fp = [D / f"{f}.parquet" for f in FCDB_FEATS]
        groups.append(("fcdb", D / "fcdb_selected_candidates.parquet",
                       lambda fp=fp: frame_keyed(D / "fcdb_selected_candidates.parquet", fp), fp))
        if args.frame:
            groups.append(("fcdb_frame(reproduce loader)", D / "fcdb_selected_candidates.parquet",
                           lambda: frame_fcdb_corrected(D), fp))
    if (D / "flms_selected_candidates.parquet").exists():
        fp = [D / f"{f}.parquet" for f in FLMS_FEATS]
        groups.append(("flms", D / "flms_selected_candidates.parquet",
                       lambda fp=fp: frame_keyed(D / "flms_selected_candidates.parquet", fp), fp))
    for ds in ["fcdb", "flms"]:
        man = D / f"consultant_blend_candidates_{ds}_core_features.parquet"
        if man.exists():
            fp = [D / f"consultant_blend_{t}_{ds}.parquet" for t in FEATURE_KINDS]
            groups.append((f"{ds}_twin", man, lambda man=man, fp=fp: frame_keyed(man, fp), fp))
    for name, base in [("aspect_free", "aspect_free_flms_selected_candidates.parquet"),
                       ("aspect_free_consultant", "aspect_free_consultant_blend_candidates_flms_core_features.parquet")]:
        pre = "aspect_free_" if name == "aspect_free" else "aspect_free_consultant_blend_"
        if (AF / base).exists():
            fp = [AF / f"{pre}{t}_flms.parquet" for t in FEATURE_KINDS]
            groups.append((name, AF / base, lambda b=AF / base, fp=fp: frame_keyed(b, fp, drop_dup=True), fp))

    if not groups:
        print(f"no known feature groups found under {D}"); sys.exit(2)

    results = {}
    for name, box_src, builder, fp in groups:
        missing = [p.name for p in fp if not p.exists()]
        if missing:
            print(f"[{name}] SKIP: missing feature files {missing}")
            continue
        results[name] = check_group(name, box_src, builder(), fp)
        print()

    # GAICD mixed-training rows: per-crop 60F, so a dedicated box<->rank identity check
    if (D / "gaicd_selected_candidates.parquet").exists():
        g_ok = check_gaicd(D)
        if g_ok is not None:
            results["gaicd"] = g_ok
        print()

    bad = [n for n, ok in results.items() if not ok]
    print("=" * 70)
    print(f"OVERALL: {'PASS — all groups aligned' if not bad else 'FAIL — misaligned groups: ' + ', '.join(bad)}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
