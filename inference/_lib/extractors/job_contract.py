"""Bind scratch files and enforce complete extractor results for warm and cold jobs."""
import functools
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_contract import validate_boxes, validate_rows

def extraction_job(bindings):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(args=None):
            import pandas as pd
            scratch = (args or {}).get("scratch") or os.environ.get("CROPPER_SCRATCH")
            if not scratch:
                raise RuntimeError("extractor requires an invocation scratch directory")
            for name, filename in bindings.items():
                fn.__globals__[name] = Path(scratch) / filename
            source = "refine_crops.parquet" if "IN" in bindings else "cands.parquet"
            expected = pd.read_parquet(Path(scratch) / source)
            validate_boxes(expected)
            result = fn(args)
            for filename in bindings.values():
                if filename == source or not filename.endswith(".parquet") or any(x in filename for x in ("embeddings", "patch_norms")):
                    continue
                validate_rows(expected, pd.read_parquet(Path(scratch) / filename), filename,
                              keys=["s"] if source.startswith("refine") else ["photo_id", "dpp_rank"])
            return result
        return wrapped
    return decorate
