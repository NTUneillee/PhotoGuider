#!/usr/bin/env python3
"""Worker pool seams: a dead or unwritable worker falls back to a cold run; release_gpu stops the
resident workers (they restart lazily) and is a no-op for cold extractors."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference" / "_lib")]
import workers  # noqa: E402
from feature_extractors import Extractors  # noqa: E402


class BrokenStdin:
    def write(self, _):
        raise BrokenPipeError

    def flush(self):
        pass


class DeadProc:
    stdin = BrokenStdin()

    def poll(self):
        return None


def main():
    calls = []
    workers.run_cold = lambda op, args: calls.append((op, args)) or True
    pool = workers.ExtractorPool()
    pool._ensure = lambda env_key: DeadProc()
    assert pool.run_op("ext_laion", {"scratch": "/x"}) is True and calls == [("ext_laion", {"scratch": "/x"})]
    assert pool.procs.get("laion") is None
    print("dead worker -> cold fallback: PASS")

    released = []
    with tempfile.TemporaryDirectory() as d:
        Extractors(d, runner=lambda op, a: True, release=lambda: released.append(1)).release_gpu()
        Extractors(d).release_gpu()                              # cold: nothing resident, no error
    assert released == [1]
    closed = workers.ExtractorPool()
    closed.procs["laion"] = "cold"
    closed.close()
    assert closed.procs == {}
    print("release_gpu hook and pool.close: PASS")


def test_extractor_input_columns():
    """Pipeline-only columns (NaN on twin rows, like the conditioned pool's iou_gt) never reach extractors."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from feature_extractors import EXTRACTOR_INPUT
    seen = {}
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "x.png"; Image.new("RGB", (64, 48)).save(img)
        frame = pd.DataFrame([dict(photo_id=0, dpp_rank=k, image_path=str(img), x1=0.0, y1=0.0, x2=32.0, y2=24.0,
                                   img_w=64.0, img_h=48.0, src="k10", iou_gt=(0.5 if k < 2 else np.nan))
                              for k in range(3)])

        def runner(op, args):
            seen[op] = list(pd.read_parquet(Path(args["scratch"]) / "cands.parquet").columns)
            return False
        try:
            Extractors(d, runner=runner).deep(frame)
        except RuntimeError:
            pass
    assert seen and all(cols == EXTRACTOR_INPUT for cols in seen.values()), seen
    print("extractors see only their input columns: PASS")


if __name__ == "__main__":
    main()
    test_extractor_input_columns()
