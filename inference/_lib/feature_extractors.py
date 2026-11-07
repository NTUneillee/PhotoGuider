"""Deep-feature extraction behind one small interface.

The per-crop deep features (CLIP-L, LAION aesthetic head, FGAesQ, SAM encoder/decoder) are computed
by scripts that each run in their own Python environment (see envs/). This module owns everything a
caller needs to know about that:

    ex = Extractors(scratch_dir)                       # one-shot subprocess per job (cold)
    ex = Extractors(scratch_dir, runner=pool.run_op)   # jobs served by a resident ExtractorPool (warm)
    frame = ex.deep(frame)                 # + the 51 deep columns, validated row-for-row
    la, fg = ex.scale_scores(rows)         # LAION / FGAesQ scores of the refinement crops, keyed by scale

``scratch_dir`` is the only place the extractors read inputs from and write outputs to; every job
is validated against the rows it was given (feature_contract), and a missing or partial output
raises with the name of the failing extractor instead of surfacing later as a KeyError.

``OP_ENV`` is the single registry of extractor script -> environment; workers.py and
worker_serve.py read it from here.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXTRACTORS = HERE / "extractors"
INF_DIR = HERE.parent

import sys  # noqa: E402
sys.path.insert(0, str(INF_DIR))
from env_config import require_env  # noqa: E402
from feature_contract import validate_boxes, validate_rows  # noqa: E402

# extractor op (script stem) -> environment that runs it
OP_ENV = {
    "ext_clip": "main",
    "ext_laion": "laion", "score_crops_laion": "laion",
    "ext_fgaesq": "fgaesq", "score_crops_fgaesq": "fgaesq",
    "ext_sam_enc": "sam", "ext_sam_dec": "sam",
}
ENV_OPS = {}
for _op, _env in OP_ENV.items():
    ENV_OPS.setdefault(_env, []).append(_op)

# the deep-feature outputs, by producing op; every one is required by the ranker
DEEP_OUTPUTS = {
    "ext_clip": ["feat_clip"],
    "ext_laion": ["feat_laion"],
    "ext_fgaesq": ["feat_fgaesq"],
    "ext_sam_enc": ["feat_sam_enc_handcrafted", "feat_sam_enc_spatial", "feat_sam_enc_mask"],
    "ext_sam_dec": ["feat_sam_dec_stage124", "feat_sam_dec_stage3", "feat_sam_dec_stage5"],
}
DEEP_INTERMEDIATE = ["feat_sam_image_embeddings", "feat_sam_cand_embeddings", "feat_sam_image_patch_norms"]
# The only candidate columns any deep extractor reads. Writing just these keeps pipeline-side columns
# (e.g. the conditioned pool's iou_gt, absent on twin rows) out of the extractors' echoed output,
# where the job contract would reject them as non-finite feature values.
EXTRACTOR_INPUT = ["photo_id", "dpp_rank", "image_path", "x1", "y1", "x2", "y2", "img_w", "img_h"]


def run_cold(op, args):
    """One-shot subprocess in the op's environment. Returns True on success; prints the tail of the
    failing process's output otherwise."""
    env_py = require_env(OP_ENV[op])
    r = subprocess.run([str(env_py), str(EXTRACTORS / f"{op}.py")], capture_output=True, text=True,
                       env=dict(os.environ, CROPPER_SCRATCH=str(args["scratch"])))
    if r.returncode != 0:
        print(f"  WARN {op}: {(r.stderr or r.stdout)[-1200:]}", flush=True)
        return False
    return True


class Extractors:
    def __init__(self, scratch, runner=None, release=None):
        self.scratch = Path(scratch)
        self.runner = runner or run_cold     # callable(op, args) -> bool
        self._release = release              # frees resident workers' GPU memory; they restart on next use

    @classmethod
    @contextmanager
    def temporary(cls, runner=None, release=None):
        """An Extractors bound to a fresh temporary scratch directory, removed on exit."""
        with tempfile.TemporaryDirectory(prefix="crop-") as d:
            yield cls(d, runner, release)

    def release_gpu(self):
        """Stop resident extractor workers before a large model (Venus, the explainer) loads on the
        same GPU; the next extraction restarts them. Cold runners hold nothing, so this is a no-op."""
        if self._release is not None:
            self._release()

    def run(self, op, args=None):
        return self.runner(op, {"scratch": str(self.scratch), **(args or {})})

    def _run_concurrently(self, ops):
        """Run ``ops`` with one thread per environment: ops of different environments overlap (each is
        its own worker process), ops sharing an environment keep their order. Returns the failed ops."""
        by_env = {}
        for op in ops:
            by_env.setdefault(OP_ENV[op], []).append(op)

        def run_chain(chain):
            failed = []
            for op in chain:
                print(f"[deep] {op} ...", flush=True)
                if not self.run(op):
                    failed.append(op)
            return failed

        # The SAM encoder transiently needs ~10 GB on a panorama's crop set; on a 16 GB card that
        # overlaps badly with the other workers starting up, so its chain runs after the others.
        sam_chain = by_env.pop("sam", [])
        with ThreadPoolExecutor(max_workers=max(1, len(by_env))) as tp:
            failed = [op for f in tp.map(run_chain, by_env.values()) for op in f]
        return failed + run_chain(sam_chain)

    # ── the 51 deep features for a candidate frame ──
    def deep(self, frame, subject_masks=None):
        """Run the five deep extractors on ``frame`` (needs photo_id, dpp_rank, image_path, box, img_w/h)
        and return the frame with their columns merged on (photo_id, dpp_rank).

        ``subject_masks``: {photo_id: (_, _, uint8 subject map)} for the frame's images — the same
        InSPyReNet map the pipeline measured. Every other mask feature is computed in-process from
        that map; the SAM encoder is the only mask consumer running in another environment, so the
        map is handed to it through the job scratch for ``sam_mask_vs_inspy_iou``. Without it that
        feature is 0 for every row."""
        validate_boxes(frame)
        validate_rows(frame, frame[["photo_id", "dpp_rank"]], "candidate input")
        outputs = [f for fs in DEEP_OUTPUTS.values() for f in fs]
        for f in outputs + DEEP_INTERMEDIATE:          # exists-skip files must not leak across images
            (self.scratch / f"{f}.parquet").unlink(missing_ok=True)
        frame[EXTRACTOR_INPUT].to_parquet(self.scratch / "cands.parquet")
        masks = self.scratch / "subject_masks.pkl"
        if subject_masks:
            pickle.dump({k: (None, None, v[2]) for k, v in subject_masks.items()}, open(masks, "wb"))
        else:
            masks.unlink(missing_ok=True)
        failed = self._run_concurrently(list(DEEP_OUTPUTS))
        absent = sorted({op for op, fs in DEEP_OUTPUTS.items()
                         if any(not (self.scratch / f"{f}.parquet").exists() for f in fs)} | set(failed))
        if absent:
            raise RuntimeError(
                f"deep feature extraction failed in: {', '.join(absent)} (its error is printed above). "
                f"The ranker needs all 60 features, so this image cannot proceed. To debug, run the "
                f"extractor directly, e.g. <env_python> {EXTRACTORS}/{absent[0]}.py with CROPPER_SCRATCH={self.scratch}")
        df = frame.copy()
        for f in outputs:
            md = pd.read_parquet(self.scratch / f"{f}.parquet")
            validate_rows(frame, md, f)
            keep = ["photo_id", "dpp_rank"] + [c for c in md.columns if c not in df.columns]
            df = df.merge(md[keep], on=["photo_id", "dpp_rank"], how="left", validate="one_to_one")
        la = pd.read_parquet(self.scratch / "feat_laion.parquet")
        if "aes" in la.columns:
            df = df.merge(la[["photo_id", "dpp_rank", "aes"]].rename(columns={"aes": "laion_score"}),
                          on=["photo_id", "dpp_rank"], how="left")
        return df

    # ── aesthetic scores of the refinement crops (expansion gate input) ──
    def scale_scores(self, rows):
        """``rows``: records with image_path, img_w/h, x1..y2 and the scale ``s`` of each crop.
        Returns two dicts s -> score (LAION, FGAesQ). Raises if either scorer fails."""
        df = pd.DataFrame(rows)
        validate_boxes(df)
        df.to_parquet(self.scratch / "refine_crops.parquet")
        failed = self._run_concurrently(["score_crops_laion", "score_crops_fgaesq"])
        if failed:
            raise RuntimeError(f"refinement feature extraction failed: {', '.join(failed)}")
        out = []
        for name in ("laion", "fgaesq"):
            got = pd.read_parquet(self.scratch / f"refine_{name}.parquet")
            validate_rows(df, got, name, keys=["s"])
            out.append({float(s): float(v) for s, v in zip(*got[["s", "score"]].values.T)})
        return out[0], out[1]
