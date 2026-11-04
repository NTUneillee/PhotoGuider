"""The single-image crop pipeline: pool -> DPP shortlist -> token ranker -> gated refinement
-> (full mode) consultant twins + agreement-gated blend.

    res, ctx = run(image, aspect, full, out, extractors)

Two protocols, chosen by ``aspect``:
  CONDITIONED (an aspect "W:H" is requested): the 8-source production pool at the requested ratio,
      Protocol CONDITIONED (st +0.03, sh 0.05, blend 0.4/0.2/tau 0.5, ratio kept).
  FREE-FORM (no aspect): the aspect-free grid pool (eight ratios x 7 scales x 36 positions, each
      candidate's own ratio is its request), Protocol FREE_FORM (st -0.02, sh 0.08, blend 0.7/0.3/
      tau 0.45, ratio adopted from the consultant).

``full`` adds the consultant: Venus is decoded live in its own environment (1 greedy + m samples
+ M paraphrases), twin candidates are built for shortlist members agreeing with the core opinion
(IoU >= 0.5), the frozen ranker compares original and twin with a fixed margin, and the refined
frame concedes to the broader opinion only above the protocol's agreement threshold.

Everything numerical is in _lib/release_model.py (one copy, pinned by the benchmark golden test);
deep features come through _lib/feature_extractors.py. All maps are computed live from the image
(DINOv3 saliency, InSPyReNet subject mask, CLIP-L composition attention); nothing here reads
benchmark caches.
"""
import sys, os, json, multiprocessing, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from PIL import Image
sys.stdout.reconfigure(line_buffering=True)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from env_config import require_env
LIB = HERE/"_lib"
MOD = HERE/"models"
sys.path.insert(0, str(LIB))
import release_model as M                                    # the decision layer: constants, geometry, ReleaseModel
from feature_extractors import Extractors                    # deep features / scale scores (also PanoCropping's INFER.Extractors)
import pool_gen as PG                                        # 8-source conditioned pool generator
import aspect_free_pool as AF1                               # aspect-free pool enumeration
import aspect_free_selection as AF2                          # canonical DPP numerics
import consultant_features as M02                            # twin core-feature extractor (canonical path)
from aspect import parse_aspect


# ─────────────── consultant (Venus) ───────────────
def consult(image_path, out_dir):
    """Decode the consultant's opinions for one image in its own environment:
    (greedy box, temperature-sampled boxes, paraphrase-prompt boxes)."""
    venus_py = require_env("venus")
    vdir = Path(out_dir) / "venus_live"; vdir.mkdir(parents=True, exist_ok=True)
    raw = vdir / "decode.json"
    print("[consultant] decoding live boxes ...", flush=True)
    r = subprocess.run([str(venus_py), str(HERE/"venus_live.py"), "--image", str(image_path),
                        "--out", str(raw)], capture_output=True, text=True)
    if r.returncode != 0 or not raw.exists():
        raise SystemExit(f"venus_live.py failed (rc={r.returncode}):\n{r.stderr[-800:]}")
    dec = json.load(open(raw))[str(image_path)]
    return dec["greedy"], dec["multibox"], dec["promptens"]


# ─────────────── maps ───────────────
def get_saliency_mask(image_path):
    """(vmix_2d, v1_2d, mask_np, rgb): live DINOv3 saliency (mixed + attention) and the InSPyReNet subject mask."""
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    import feat_saliency as FS
    print("[saliency] generating live (DINOv3 attn+PCA + InSPyReNet)")
    vmix, v1d = FS.build_saliency(image_path); mk = FS.build_mask(image_path)
    # release the live models' GPU memory before the extractor subprocesses run (they need the
    # GPU too; keeping ~12GB of saliency-model cache resident OOMs ext_sam_* on 16GB cards)
    try:
        FS._STATE.clear()
        import gc, torch as _t; gc.collect(); _t.cuda.empty_cache()
    except Exception:
        pass
    return vmix, v1d, mk, rgb


def maps_for(pid_key, vmix_2d, v1_2d, mask_np):
    """The (vmix, v1, mask) map triple the core featurizers take, keyed by this image's id."""
    return ({pid_key: (None, None, vmix_2d)}, {pid_key: (None, None, v1_2d)}, {pid_key: (None, None, mask_np)})


_FEAT_WORKERS = int(os.environ.get("PHOTOGUIDER_WORKERS") or max(1, min(8, (os.cpu_count() or 2) - 1)))
_FEAT_POOL = None


def _chunk_features(args):
    return M02.stage_c_features(args[0].reset_index(drop=True), args[1])


def af_chain_features(dfp, vmix_2d, v1_2d, mask_np):
    """Core-feature family via the canonical code path (free-form pool and twins alike) on the
    in-memory maps -> numerics identical to the benchmark chains. The per-row loop is independent
    across rows, so a large pool is split across forked CPU workers (numpy only, never CUDA) and
    concatenated back in order; set PHOTOGUIDER_WORKERS=1 for the serial path."""
    global _FEAT_POOL
    maps = maps_for(dfp.photo_id.iloc[0], vmix_2d, v1_2d, mask_np)
    dfp = dfp.reset_index(drop=True)
    if len(dfp) < 200 or _FEAT_WORKERS == 1:
        return M02.stage_c_features(dfp, maps)
    if _FEAT_POOL is None:
        _FEAT_POOL = multiprocessing.get_context("fork").Pool(_FEAT_WORKERS)
    bounds = [round(i * len(dfp) / _FEAT_WORKERS) for i in range(_FEAT_WORKERS + 1)]
    out = pd.concat(_FEAT_POOL.map(_chunk_features, [(dfp.iloc[a:b], maps) for a, b in zip(bounds, bounds[1:]) if b > a]),
                    ignore_index=True)
    assert len(out) == len(dfp), "parallel candidate features lost rows"
    return out


# ─────────────── candidate pool (per protocol) ───────────────
def gen_pool_ff(W, H, vmix_2d, v1_2d, mask_np, image_path):
    """Free-form pool: aspects = AF1.enum_aspects (7 fixed + original, 2% dedupe) x 7 scales x 36 positions;
    gt_aspect := the candidate's own enumerated ratio ("requested ratio" semantics)."""
    Wi, Hi = int(W), int(H)
    rows, seen = [], set()
    for asp in AF1.enum_aspects(Wi, Hi):
        for scale in AF1.SCALES:
            if asp >= Wi/Hi:
                cw = int(Wi*scale); ch = int(cw/asp)
            else:
                ch = int(Hi*scale); cw = int(ch*asp)
            if cw < 50 or ch < 50 or cw > Wi or ch > Hi: continue
            for cx_f, cy_f in AF1.POSITIONS:
                cx = cx_f*Wi; cy = cy_f*Hi
                x1 = max(0, int(cx - cw/2)); y1 = max(0, int(cy - ch/2))
                x2 = min(Wi, x1+cw); y2 = min(Hi, y1+ch)
                if x2-x1 < 50 or y2-y1 < 50: continue
                key = (x1, y1, x2, y2)
                if key in seen: continue
                seen.add(key)
                asp_bin = int(round(asp*10))/10
                rows.append({"photo_id": 0, "image_path": str(image_path),
                             "src": f"G_a{asp_bin:.1f}_s{scale:.2f}", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "img_w": Wi, "img_h": Hi, "gt_aspect": asp})
    return af_chain_features(pd.DataFrame(rows), vmix_2d, v1_2d, mask_np)


def gen_pool(W, H, target_aspect, vmix_2d, v1_2d, mask_np, rgb, image_path):
    if target_aspect is not None:
        return PG.generate_pool(None, W, H, target_aspect, vmix_2d, v1_2d, mask_np, rgb, image_path)
    return gen_pool_ff(W, H, vmix_2d, v1_2d, mask_np, image_path)


def run_dpp_k10(df, target_aspect):
    """DPP shortlist of ten: quality model + saliency boost + aspect penalty, greedy DPP."""
    dq = joblib.load(MOD/"dpp_quality.pkl")
    dppm, F = dq["model"], list(dq["features"])
    meta = joblib.load(MOD/"dpp_meta.pkl")
    df = df.copy().reset_index(drop=True); df["src"] = df["src"].astype(str)
    if target_aspect is not None:
        df["gt_aspect"] = target_aspect                 # conditioned: aspect factor uses the request
    with M.lgbm_quiet():
        q_preds = np.clip(dppm.predict(df[F].values), 1e-6, None)
    q = AF2.make_q_dpp(df, q_preds, meta["vmix_boost"], meta["aspect_alpha"])
    sel = AF2.run_dpp_per_image(df, q, K=10, beta=meta["dpp_beta"], src_penalty=meta["src_penalty"])
    idx = sel[df.photo_id.iloc[0]]
    k10 = df.loc[idx].copy(); k10["dpp_rank"] = list(range(len(idx)))
    return k10.reset_index(drop=True)


def build_twins(k10, v6, W, H, vmix_2d, v1_2d, mask_np):
    """Twin candidates: for every shortlist candidate c with IoU(c, v6) >= 0.5, twin = make_twin(c, v6)
    at dpp_rank = 10 + base rank, featurized on the canonical twin path."""
    rows = []
    for r in k10.itertuples(index=False):
        cand = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))
        if M.iou_xyxy(cand, v6) < M.TWIN_IOU: continue
        b = M.make_twin(cand, v6, W, H)
        rows.append(dict(photo_id=r.photo_id, image_path=r.image_path, src="twin",
                         x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                         img_w=float(W), img_h=float(H), gt_aspect=float(r.gt_aspect),
                         dpp_rank=10 + int(r.dpp_rank)))
    man = pd.DataFrame(rows)
    if not len(man): return man
    feats = af_chain_features(man, vmix_2d, v1_2d, mask_np)
    out = pd.concat([man.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


with_pi = M.with_pi   # per-frame laion_pi_minmax + 60-feature completeness check


# ─────────────── refinement inputs ───────────────
def aes_scale_deltas(extractors, image_path, rc, W, H):
    """LAION+FGAesQ on {expand_s(rc, s)} -> delta_aes_ens(s): the change of the aesthetic scores with
    scale, z-scored against frozen FCDB-train statistics (single-image-safe surrogate for the
    per-group transductive z-scoring of the benchmark). Returns {s: delta} for s in M.SS."""
    ref = joblib.load(MOD/"aes_ref_stats.pkl")
    rows = [{"image_path": str(image_path), "img_w": float(W), "img_h": float(H),
             "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "s": float(s)}
            for s in [1.0]+list(M.SS) for b in [M.expand_s(rc, s, W, H) if s > 1.0 else rc]]
    la, fg = extractors.scale_scores(rows)
    out = {}
    for s in M.SS:
        mu_l, sd_l = ref["laion"][s]; mu_f, sd_f = ref["fgaesq"][s]
        zl = (la[s]-la[1.0] - mu_l)/(sd_l+1e-9); zf = (fg[s]-fg[1.0] - mu_f)/(sd_f+1e-9)
        out[s] = zf + zl
    return out


def load_refine_ctx(image_path):
    """Live CLIP-L composition attention (CAM) + composition-type vector for the recenter / comp gates."""
    print("[refine-ctx] live CLIP-L CAM + composition-type ...", flush=True)
    import feat_clip_refine as FCR
    heat, ct_vec = FCR.cam_and_ct(image_path)
    return {"heat": heat, "ct": {"__live__": ct_vec}}


def backhalf(t, crop, mask, W, H, protocol, RC, model, image_path, extractors):
    """Gated recenter + expansion gates (MAX-union over coverage / composition / golden / aesthetics)
    + shrink fallback, through the release model."""
    th = M.thirds_snap(M.salient_center(mask)); cam = M.cam_center(RC["heat"])
    gf = M.gate_feats(t, th, cam); w = model.recenter_weight(gf)
    rc = model.recenter(crop, gf, th, cam, W, H)
    ssr = float(t.subject_size_ratio); samcos = float(t.sam_cos_to_image)
    pid_k, grp, ct = "__live__", "inf", RC["ct"]
    g0 = float(M02.golden_align((rc[0]+rc[2])/2/W, (rc[1]+rc[3])/2/H))
    glut = {("inf", str(pid_k), float(s)): float(M02.golden_align(
        (M.expand_s(rc, s, W, H)[0]+M.expand_s(rc, s, W, H)[2])/2/W,
        (M.expand_s(rc, s, W, H)[1]+M.expand_s(rc, s, W, H)[3])/2/H)) - g0 for s in M.SS}
    print("[refine] live LAION+FGAesQ aes scale-gate ...", flush=True)
    aes_d = aes_scale_deltas(extractors, image_path, rc, W, H)
    alut = {("inf", str(pid_k), float(s)): aes_d[s] for s in M.SS}
    fc, scale_used, _ = model.scale_box(rc, W, H, protocol, mask, ssr, samcos, ct, alut, glut, grp, pid_k)
    return fc, {"pan_w": w, "scale": scale_used}


# ─────────────── the pipeline ───────────────
def run(image, aspect, full, out, extractors):
    """Compute the best crop for one image; return (result_dict, context).

    ``aspect``: "W:H" / number / None (validated by the caller).  ``full``: add the consultant
    (decoded live into ``out``).  ``extractors``: a feature_extractors.Extractors bound to a scratch
    directory (and, optionally, a resident worker pool).
    ``context`` carries the maps and geometry an embedding caller reuses for the assessments."""
    t0 = time.time()
    ta = parse_aspect(aspect) if aspect else None
    cond = ta is not None
    P = M.protocol_for(ta)
    img = Image.open(image); W, H = float(img.size[0]), float(img.size[1])
    print(f"[mode] protocol={'CONDITIONED' if cond else 'FREE-FORM'} ({'aspect='+f'{ta:.4f}' if cond else 'aspect-free'}) "
          f"| {'FULL(consultant)' if full else 'FAST'} | st={P.st:+.2f} sh={P.sh:.2f}"
          f"{f' | twins(delta={P.twin_delta},iou>={M.TWIN_IOU}) + blend(bc={P.bc},ba={P.ba},tau={P.tau},br={P.br})' if full else ''}")
    print(f"[infer] image {image} ({int(W)}x{int(H)})")
    vmix, v1, mask, rgb = get_saliency_mask(image)
    pool = gen_pool(W, H, ta, vmix, v1, mask, rgb, image)
    print(f"[pool] {len(pool)} candidates ({'8-src' if cond else 'af-grid'})")
    k10 = run_dpp_k10(pool, ta)
    print("[dpp] K=10 selected (canonical quality model)")

    # FULL: the consultant's opinions and the twin candidates
    twins = pd.DataFrame(); v6 = g = mb = pe = None
    if full:
        extractors.release_gpu()                 # Venus needs ~10 GB: resident workers step aside first
        g, mb, pe = consult(image, out)
        v6 = M.consensus_box(g, mb)
        twins = build_twins(k10, v6, W, H, vmix, v1, mask)
        print(f"[twins] {len(twins)} twins (v6={tuple(round(x,1) for x in v6)})")

    frame = pd.concat([k10, twins], ignore_index=True, sort=False) if len(twins) else k10
    frame = extractors.deep(frame, subject_masks={frame.photo_id.iloc[0]: (None, None, mask)})
    miss = [c for c in M.FC if c not in frame.columns and c != "laion_pi_minmax"]   # pi is recomputed per frame
    if miss:
        raise RuntimeError(f"ranker features missing after deep extraction: {miss} — "
                           f"an extractor produced an incomplete parquet (see [deep] output above)")

    # the release decision layer (ranker ensemble + gates) + optional twin isolated flip.
    # K3 SELECTION (= reproduce.py): laion_pi_minmax is computed over the FULL K10(+twin) frame,
    # then the rows are truncated to DPP top-3 and the ranker's per-image normalization runs on them.
    model = M.ReleaseModel.load(MOD); score_fn = model.score
    k10d = with_pi(frame[frame.dpp_rank < 10])
    k3d = k10d[k10d.dpp_rank < P.k_sel].reset_index(drop=True)
    tf = k3d.loc[int(np.argmax(score_fn(k3d)))]
    crop_fast = (float(tf.x1), float(tf.y1), float(tf.x2), float(tf.y2))
    print(f"[ranker] fast top-1 crop {tuple(round(x,1) for x in crop_fast)} "
          f"(rank {int(tf.dpp_rank)} of top-{P.k_sel}, {tf.src})")
    winner, flipped = tf, False
    if full and len(twins):
        mf = with_pi(frame)                                   # pi over the full merged set (benchmark)
        mf = mf[(mf.dpp_rank < P.k_sel) | (mf.dpp_rank >= 10)].reset_index(drop=True)
        qv = model.score_with_twins(mf, P.twin_delta)
        wi = int(np.argmax(qv)); winner = mf.loc[wi]; flipped = bool(winner.dpp_rank >= 10)
        print(f"[twin-flip] winner rank {int(winner.dpp_rank)} ({'FLIPPED to twin' if flipped else 'no flip'})")
    crop_win = (float(winner.x1), float(winner.y1), float(winner.x2), float(winner.y2))

    # refinement — fast winner; full winner if different
    RC = load_refine_ctx(image)
    fc_fast, rinfo = backhalf(tf, crop_fast, mask, W, H, P, RC, model, image, extractors)
    fc_full = fc_fast
    if full and crop_win != crop_fast:
        fc_full, _ = backhalf(winner, crop_win, mask, W, H, P, RC, model, image, extractors)
    print(f"[refine] fast final {tuple(round(x,1) for x in fc_fast)} (pan_w={rinfo['pan_w']:.2f} scale={rinfo['scale']})")

    # FULL: agreement-gated concession to the broader opinion
    final_full = fc_full
    if full:
        vf = M.expanded_opinion(g, mb, pe, W, H, degen=P.degen)
        final_full, blended = M.blend(fc_full, vf, P, W, H)
        print(f"[consultant] v_f={tuple(round(x,1) for x in vf)} agree={'YES(blend)' if blended else 'NO(keep)'} "
              f"-> full final {tuple(round(x,1) for x in final_full)}")
    final = final_full if full else fc_fast
    print(f"[final] {'FULL' if full else 'FAST'} crop {tuple(round(x,1) for x in final)}  ({time.time()-t0:.0f}s)")

    res = {"image": str(image), "mode": "cond" if cond else "ff",
           "k10": [{"dpp_rank": int(r.dpp_rank), "src": str(r.src),
                    "box": [float(r.x1), float(r.y1), float(r.x2), float(r.y2)]} for r in k10d.itertuples(index=False)],
           "fast": {"top1": list(crop_fast), "final": list(fc_fast)},
           "full": {"winner": list(crop_win), "final": list(final_full)} if full else None}
    ctx = dict(vmix=vmix, v1=v1, mask=mask, W=W, H=H)
    return res, ctx
