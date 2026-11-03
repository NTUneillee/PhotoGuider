"""Benchmark reproduction from the SHIPPED model artifacts — no training, pure CPU, ~25 s.

Loads the decision layer from inference/models/ (ReleaseModel), reads the frozen feature caches
of both benchmarks, runs selection -> refinement -> (full mode) consultant blend, and checks the
four cells of paper Table 1 against EXPECT at tolerance TOL:

  CONDITIONED (FCDB test 345, GT aspect as the request):  full 0.7660 / 0.0559   fast 0.7593 / 0.0584
  FREE-FORM   (FLMS 500, aspect-free):                    full 0.8563 / 0.0328   fast 0.8346 / 0.0412

The per-image boxes it writes (boxes_<cell>.csv) are the same boxes the cell means are computed
from; tests/test_reproduce_golden.py compares them with the tracked inference/results/ files.

The fast row is the full row minus the consultant stages, never separately tuned. FLMS is strictly
zero-shot. The consultant boxes are frozen artifacts (no MLLM runs here). The ranker was trained
on FCDB-train 1,369 + GAICD-train 1,000 images and every gate on FCDB-train only; training is not
part of this package.

Usage: python inference/reproduce.py [--data_dir DIR] [--output-dir DIR]   (default DIR: <release>/caches)
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_lib"))
from release_model import (FC, CONDITIONED, FREE_FORM, ReleaseModel, per_img_minmax, mi, thirds_snap,  # noqa: E402
                           salient_center, cam_center, gate_feats, expanded_opinion, blend)

# expected values (paper Table 1; K3 selection)
EXPECT = dict(cond_full=(0.765970, 0.055899, 218), cond_fast=(0.759273, 0.058407, 210),
              ff_full=(0.856323, 0.032806, 468),  ff_fast=(0.834630, 0.041183, 424))
TOL = 0.0005

CONSULTANT_FEATURE_FILES = ["consultant_blend_laion_{ds}.parquet",
                            "consultant_blend_clipmp_{ds}.parquet",
                            "consultant_blend_fgaesq_{ds}.parquet",
                            "consultant_blend_segment_encoder_{ds}.parquet",
                            "consultant_blend_segment_decoder_{ds}.parquet"]


def k_sel(df, k):
    """eval-pool truncation: keep DPP top-k + twin consultant rows (dpp_rank >= 10)."""
    return df[(df.dpp_rank < k) | (df.dpp_rank >= 10)].reset_index(drop=True)


# ════════════════ loaders (FCDB = box-coord key join, alignment-audited) ════════════════
FCDB_FEATS = ["fcdb_laion_aesthetic_scores","handcrafted_geometry_features","spatial_composition_features","subject_mask_features",
              "subject_composition_integrity_features","subject_composition_features","aesthetic_integrity_features","fgaesq_features",
              "clip_multi_prompt_features"]

def box_key(df, nd=3):
    """(photo_id, rounded box coords) — the physical identity of a candidate row."""
    return list(zip(df.photo_id.tolist(),
                    np.round(df.x1.values, nd), np.round(df.y1.values, nd),
                    np.round(df.x2.values, nd), np.round(df.y2.values, nd)))

def _fcdb_frame(D):
    """FCDB loader (merge steps only; no fillna, so missing values stay visible to the self-check).
    dpp_rank is re-derived from the candidate parquet's own box<->rank assignment via rounded
    box coords, rows are stable-sorted, and every deep-feature parquet is merged strictly
    key-based on (photo_id, dpp_rank). No sort-order-implied keying anywhere."""
    base = pd.read_parquet(D/"fcdb_selected_candidates.parquet")
    tmap = dict(zip(box_key(base), base.dpp_rank.tolist()))
    assert len(tmap) == len(base), "duplicate rounded boxes in candidate parquet"
    base = base.copy()
    base["dpp_rank"] = [tmap[k] for k in box_key(base)]      # box-coord-derived rank
    base = base.sort_values(["photo_id","dpp_rank"], kind="mergesort").reset_index(drop=True)
    for fn in FCDB_FEATS:
        f = pd.read_parquet(D/f"{fn}.parquet")
        clash = [c for c in f.columns if c in base.columns and c not in ("photo_id","dpp_rank")]
        assert not clash, f"{fn}: column collision {clash}"
        base = base.merge(f, on=["photo_id","dpp_rank"], how="left")
    return base

def load_fcdb(D):
    return _fcdb_frame(D).fillna(0.0)

def alignment_self_check(D):
    """Integrity gate on the cache archive: rebuilds the FCDB frame WITHOUT fillna and verifies, for
    every row and every feature parquet, that the frame's dpp_rank equals the box-coord-derived true
    rank and that the feature values on the row equal the parquet's values at that rank (NaN==NaN).
    Returns True iff the frame prints ALIGNED."""
    frame = _fcdb_frame(D)
    truth = pd.read_parquet(D/"fcdb_selected_candidates.parquet")
    tmap = dict(zip(box_key(truth), truth.dpp_rank.tolist()))
    true_rank = np.array([tmap[k] for k in box_key(frame)])
    key_bad = int((frame.dpp_rank.values != true_rank).sum())
    print(f"[self-check] fcdb frame rows {len(frame)}, pids {frame.photo_id.nunique()}, "
          f"key mismatches (frame dpp_rank vs box-derived rank): {key_bad}", flush=True)
    ok = key_bad == 0
    idx = frame[["photo_id"]].copy(); idx["dpp_rank"] = true_rank
    for fn in FCDB_FEATS:
        f = pd.read_parquet(D/f"{fn}.parquet")
        cols = [c for c in f.columns if c not in ("photo_id","dpp_rank")]
        ref = idx.merge(f, on=["photo_id","dpp_rank"], how="left")
        bad = np.zeros(len(frame), dtype=bool)
        for c in cols:
            a, b = frame[c].values, ref[c].values
            bad |= ~((a == b) | (pd.isna(a) & pd.isna(b)))
        n_bad = int(bad.sum()); ok &= n_bad == 0
        print(f"[self-check]   {fn:32s} {n_bad:6d}/{len(frame)} rows misaligned  "
              f"{'OK' if n_bad == 0 else 'FAIL'}", flush=True)
    print(f"[self-check] FCDB frame VERDICT: {'ALIGNED' if ok else 'MISALIGNED'}", flush=True)
    return ok

def load_flms(D):
    df = pd.read_parquet(D/"flms_selected_candidates.parquet")
    for fn in ["flms_laion_scores","flms_handcrafted_geometry_features","flms_spatial_composition_features",
               "flms_subject_mask_features","flms_subject_composition_integrity_features","flms_subject_composition_features",
               "flms_aesthetic_integrity_features","flms_fgaesq_scores","flms_clip_multi_prompt_features"]:
        df = df.merge(pd.read_parquet(D/f"{fn}.parquet"), on=["photo_id","dpp_rank"], how="left")
    return df.fillna(0.0)

def load_fcdb_gt(D):
    gt = {}
    for fn in ["FCDB-train.json","FCDB-test.json"]:
        for d in json.load(open(D/fn)):
            x,y,w,h = d["crop"]; gt[int(d["flickr_photo_id"])] = (float(x),float(y),float(x+w),float(y+h))
    return gt

def load_flms_gt(D):
    path = D / "flms_500_gt.mat"
    if not path.is_file():
        raise SystemExit(
            "FLMS ground truth is user-supplied and is not in the cache archive. "
            "Download release_data.tar from the official FLMS project source "
            "(http://fangchen.org/proj_page/FLMS_mm14/), then place flms_500_gt.mat at "
            f"{path}. See tools/FLMS_GT.md.")
    m = sio.loadmat(path); arr = m["img_gt"]; gt = {}
    for i in range(arr.shape[0]):
        fn = str(arr[i,0]["filename"][0]); v = []
        for g in arr[i,0]["bbox"]:
            a,b,c,d = map(float, (g[0],g[1],g[2],g[3]))
            if a < 0 or b < 0 or c <= a or d <= b: continue
            v.append((b,a,d,c))
        gt[fn] = v
    return gt

def load_heat(npz):
    z = np.load(npz, allow_pickle=True)
    return {(int(p) if str(p).isdigit() else str(p)): h for p, h in zip(z["pids"], z["heatmaps"])}

def load_aes_lut(D):
    a = pd.read_parquet(D/"aes_delta_ens.parquet")
    return {(r.group, str(r.pid), float(r.s)): float(r.delta_aes_ens) for r in a.itertuples(index=False)}

def load_gold_lut(D):
    a = pd.read_parquet(D/"aes_gold_delta.parquet")
    return {(r.group, str(r.pid), float(r.s)): float(r.delta_golden) for r in a.itertuples(index=False)}


# ════════════ twin-candidate pool (frozen consultant artifacts) ════════════
def load_twins(D, ds):
    """Twin manifest (core-feature family) + 5 re-extracted content-feature parquets, joined on
    (photo_id, dpp_rank). dpp_rank>=10 marks twin rows."""
    man = pd.read_parquet(D / f"consultant_blend_candidates_{ds}_core_features.parquet")
    for pattern in CONSULTANT_FEATURE_FILES:
        man = man.merge(pd.read_parquet(D / pattern.format(ds=ds)), on=["photo_id", "dpp_rank"], how="left")
    chk = [c for c in FC if c != "laion_pi_minmax"]  # pi_minmax recomputed at merge
    n_nan = man[chk].isna().sum().sum()
    assert n_nan == 0, f"twin {ds}: {n_nan} NaNs in FC features after join"
    return man

def merge_pool(base, twin):
    """K10 base (dpp_rank 0-9) + twin rows (dpp_rank>=10); stable sort keeps order;
    per-pid laion_pi_minmax recomputed over the merged candidate set. Missing-twin pids keep K10."""
    m = pd.concat([base, twin], ignore_index=True, sort=False)
    m = m.sort_values("photo_id", kind="mergesort").reset_index(drop=True)
    m["laion_pi_minmax"] = per_img_minmax(m, "laion_score")
    m[FC] = m[FC].fillna(0.0)
    return m.fillna({c: 0.0 for c in m.columns if m[c].dtype != object})


# ════════════════ aspect-free FLMS frame (fair free-form protocol) ════════════════
def load_af_frame(AF, consultant, base_parquet):
    df = pd.read_parquet(AF / base_parquet)
    for tag in ["laion", "clipmp", "fgaesq", "segment_encoder", "segment_decoder"]:
        middle = "consultant_blend_" if consultant else ""
        f = pd.read_parquet(AF / f"aspect_free_{middle}{tag}_flms.parquet")
        drop = [c for c in f.columns if c in df.columns and c not in ("photo_id", "dpp_rank")]
        df = df.merge(f.drop(columns=drop), on=["photo_id", "dpp_rank"], how="left")
    def pmm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)
    df["laion_pi_minmax"] = df.groupby("photo_id")["laion_score"].transform(pmm)
    df["fgaesq_rank_K10"] = (df.groupby("photo_id")["fgaesq_score"].rank(method="min", ascending=False).astype(int) - 1)
    df["fgaesq_pct_K10"] = df.groupby("photo_id")["fgaesq_score"].transform(pmm)
    df["sam_pid_norm_rank"] = (df.groupby("photo_id")["sam_emb_norm"].rank(method="min", ascending=False).astype(int) - 1)
    return df

def align_af(af10, aftw, flms_ref):
    """Column-set / dtype / order parity with the canonical FLMS frame."""
    need = [c for c in flms_ref.columns if c not in af10.columns]
    assert not need, f"missing cols in af frame: {need[:10]}"
    for df_ in (af10, aftw):
        for c in flms_ref.columns:
            if df_[c].dtype != flms_ref[c].dtype and c not in ("photo_id",):
                try: df_[c] = df_[c].astype(flms_ref[c].dtype)
                except Exception: pass
    af10 = af10[flms_ref.columns.tolist() + [c for c in af10.columns if c not in flms_ref.columns]]
    n_nan = af10[[c for c in flms_ref.columns if af10[c].dtype.kind == "f"]].isna().sum().sum()
    print(f"[frame] af10 {af10.shape} twins {aftw.shape} nan(feat) {n_nan}", flush=True)
    return af10, aftw


# ════════════════ selection + refinement + evaluation ════════════════
def select(score_fn, df, masks, heat, gt, multi):
    """Per image: the top-scoring candidate row plus everything refinement needs about it."""
    d = df.copy(); d["q"] = score_fn(df); out = []
    for pid, s in d.groupby("photo_id"):
        if pid not in gt or (multi and not gt[pid]): continue
        t = s.loc[s["q"].idxmax()]; W, H = float(t.img_w), float(t.img_h)
        crop = (float(t.x1), float(t.y1), float(t.x2), float(t.y2))
        mk = np.asarray(masks[pid][2]).astype(np.float32) if pid in masks else None
        th = thirds_snap(salient_center(mk)); cam = cam_center(heat.get(pid))
        out.append((pid, crop, W, H, gate_feats(t, th, cam), th, cam, float(t.subject_size_ratio), mk, gt[pid], float(t.sam_cos_to_image)))
    return out

def refine_all(model, tab, protocol, grp, ct, alut, glut):
    """Recenter + scale every selected crop. Returns per-image (pid, box, G, W, H)."""
    recs = []
    for pid, c, W, H, gf, th, cam, ssr, mk, G, samcos in tab:
        rc = model.recenter(c, gf, th, cam, W, H)
        fc, _, _ = model.scale_box(rc, W, H, protocol, mk, ssr, samcos, ct, alut, glut, grp, pid)
        recs.append((pid, fc, G, W, H))
    return recs

def opinions(vd, mb, pe, recs, protocol):
    """Expanded consultant opinion per image id (frozen artifacts)."""
    wh = {str(p): (W, H) for p, fc, G, W, H in recs}
    return {k: list(expanded_opinion(g, mb.get(k, []), pe.get(k, []), *wh.get(k, (1e9, 1e9)), degen=protocol.degen))
            for k, g in vd.items()}

def bde1(p, g, W, H): return (abs(p[0]-g[0])/W+abs(p[2]-g[2])/W+abs(p[1]-g[1])/H+abs(p[3]-g[3])/H)/4.0
def bde(p, G, W, H, m): return min(bde1(p, g, W, H) for g in (G if m else [G]))

def metrics(recs, vd, multi, protocol):
    """Cell metrics; vd=None -> fast row (no blend). Returns (mIoU, BDE, Acc@0.75, rows) where rows
    hold the SAME per-image boxes/values the means are computed from."""
    mis, bds, rows = [], [], []
    for pid, fc, G, W, H in recs:
        Vb = vd.get(str(pid)) if vd is not None else None
        b, _ = blend(fc, Vb, protocol, W, H) if Vb is not None else (fc, False)
        m_i = mi(b, G, multi); b_i = bde(b, G, W, H, multi)
        mis.append(m_i); bds.append(b_i)
        rows.append(dict(id=pid, x1=round(float(b[0]), 2), y1=round(float(b[1]), 2),
                         x2=round(float(b[2]), 2), y2=round(float(b[3]), 2), iou=float(m_i), bde=float(b_i)))
    return float(np.mean(mis)), float(np.mean(bds)), sum(v >= 0.75 for v in mis), rows


def evaluate(models_dir, D):
    """Run the four cells from the shipped artifacts. Returns {cell: (mIoU, BDE, Acc@0.75, rows)}."""
    D = Path(D); AF = D / "aspect_free_cache"
    if not alignment_self_check(D):
        raise SystemExit("ABORT: FCDB frame failed the alignment self-check")
    model = ReleaseModel.load(models_dir)
    fcdb = load_fcdb(D); flms_ref = load_flms(D)
    te = {int(d["flickr_photo_id"]) for d in json.load(open(D/"FCDB-test.json"))}
    fmask = pickle.load(open(D/"inspy_masks_fcdb.pkl","rb")); lmask = pickle.load(open(D/"flms_inspy_masks.pkl","rb"))
    fheat = load_heat(D/"fcdb_clip_cam_heatmaps.npz"); lheat = load_heat(D/"flms_clip_cam_heatmaps.npz")
    fgt, lgt = load_fcdb_gt(D), load_flms_gt(D)
    ct = pickle.load(open(D/"clip_composition_type.pkl","rb"))
    alut = load_aes_lut(D); glut = load_gold_lut(D)
    vte = json.load(open(D/"venus_boxes_fcdb.json")); vfl = json.load(open(D/"venus_boxes_flms.json"))
    mb_te = json.load(open(D/"venus_multibox_fcdb_te.json")); mb_fl = json.load(open(D/"venus_multibox_flms.json"))
    pe_te = json.load(open(D/"venus_promptens_fcdb_te.json")); pe_fl = json.load(open(D/"venus_promptens_flms.json"))
    sub = lambda df, ids: df[df.photo_id.isin(ids)].reset_index(drop=True)
    df_te = sub(fcdb, te)
    qd = model.score_with_twins

    cells = {}
    # ═══ CONDITIONED (FCDB) ═══
    P = CONDITIONED
    fm = merge_pool(fcdb, load_twins(D, "fcdb"))
    recs_cm = refine_all(model, select(qd, k_sel(sub(fm, te), P.k_sel), fmask, fheat, fgt, False), P, "fcdb_te", ct, alut, glut)
    cells["cond_full"] = metrics(recs_cm, opinions(vte, mb_te, pe_te, recs_cm, P), False, P)
    recs_cp = refine_all(model, select(model.score, k_sel(df_te, P.k_sel), fmask, fheat, fgt, False), P, "fcdb_te", ct, alut, glut)
    cells["cond_fast"] = metrics(recs_cp, None, False, P)
    print(f"[cond] full FCDB {cells['cond_full'][0]:.4f}/{cells['cond_full'][1]:.4f}/{cells['cond_full'][2]}/345  "
          f"fast {cells['cond_fast'][0]:.4f}/{cells['cond_fast'][1]:.4f}/{cells['cond_fast'][2]}/345", flush=True)

    # ═══ FREE-FORM (FLMS) ═══
    P = FREE_FORM
    af10 = load_af_frame(AF, False, "aspect_free_flms_selected_candidates.parquet")
    aftw = load_af_frame(AF, True, "aspect_free_consultant_blend_candidates_flms_core_features.parquet")
    af10, aftw = align_af(af10, aftw, flms_ref)
    afm = merge_pool(af10, aftw)
    recs_fp = refine_all(model, select(model.score, k_sel(af10, P.k_sel), lmask, lheat, lgt, True), P, "flms", ct, alut, glut)
    cells["ff_fast"] = metrics(recs_fp, None, True, P)
    recs_fm = refine_all(model, select(qd, k_sel(afm, P.k_sel), lmask, lheat, lgt, True), P, "flms", ct, alut, glut)
    cells["ff_full"] = metrics(recs_fm, opinions(vfl, mb_fl, pe_fl, recs_fm, P), True, P)
    print(f"[ff]   full FLMS {cells['ff_full'][0]:.4f}/{cells['ff_full'][1]:.4f}/{cells['ff_full'][2]}/500  "
          f"fast {cells['ff_fast'][0]:.4f}/{cells['ff_fast'][1]:.4f}/{cells['ff_fast'][2]}/500", flush=True)
    return cells


def main():
    ap = argparse.ArgumentParser(description="Four-cell benchmark reproduction from the shipped model artifacts.")
    ap.add_argument("--data_dir", default=str(HERE.parents[0] / "caches"),
                    help="cache mount holding every reproduction input (default: <release>/caches)")
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--output-dir", default=str(HERE / "results_reproduced"))
    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == (HERE / "results").resolve():
        ap.error("output directory must differ from tracked results")
    cells = evaluate(args.models, args.data_dir)

    order = [("cond_full", "FCDB"), ("cond_fast", "FCDB"), ("ff_full", "FLMS"), ("ff_fast", "FLMS")]
    print("="*86)
    print("FOUR-CELL REPRODUCTION FROM SHIPPED ARTIFACTS (K3 selection, twins preserved)")
    print(f"  {'cell':10s} {'bench':>5s} {'mIoU':>8s} {'BDE':>8s} {'Acc':>7s} {'exp_mIoU':>9s} {'exp_BDE':>8s} {'exp_Acc':>7s}  verdict")
    all_ok, table = True, []
    for name, bench in order:
        v_iou, v_bde, v_acc, _ = cells[name]; e_iou, e_bde, e_acc = EXPECT[name]
        ok = abs(v_iou - e_iou) <= TOL and abs(v_bde - e_bde) <= TOL and v_acc == e_acc
        all_ok &= ok
        print(f"  {name:10s} {bench:>5s} {v_iou:8.4f} {v_bde:8.4f} {v_acc:7d} {e_iou:9.4f} {e_bde:8.4f} {e_acc:7d}  {'PASS' if ok else 'FAIL'}")
        table.append(dict(cell=name, bench=bench, mIoU=v_iou, BDE=v_bde, Acc075=v_acc,
                          exp_mIoU=e_iou, exp_BDE=e_bde, exp_Acc075=e_acc, verdict="PASS" if ok else "FAIL"))
    print("-"*86)
    print(f"  Reproduction: {'OK — all four cells PASS' if all_ok else 'FAIL'} (tol {TOL})")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(output_dir/"result.csv", index=False)
    for name, _ in order:
        pd.DataFrame(cells[name][3]).to_csv(output_dir/f"boxes_{name}.csv", index=False)
        print(f"[boxes] {output_dir}/boxes_{name}.csv ({len(cells[name][3])} rows)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
