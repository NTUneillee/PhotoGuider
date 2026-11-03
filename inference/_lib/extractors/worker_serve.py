"""Long-lived extractor worker (one per environment).

Loads its environment's model(s) once, then serves per-image extraction jobs over a
line-based JSON protocol on stdin/stdout:

    request  (stdin) : one JSON object per line, {"op": <name>, "args": {...} | null}
    response (stdout): a single ACK line, @@ACK@@{"ok": true} or {"ok": false, "error": ...}

Send {"op": "__quit__"} to stop. All extractor chatter is redirected away from stdout,
so the parent reads only ACK lines. Started as:  python worker_serve.py --env <name>
"""
import argparse
import gc
import importlib
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Reduce allocator fragmentation so several resident workers share the GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _free_gpu():
    """Return reserved-but-idle GPU memory to the driver between jobs, keeping only the
    resident model weights so multiple environment workers coexist on one GPU."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

ACK = "@@ACK@@"
sys.path.insert(0, str(HERE.parent))
from feature_extractors import ENV_OPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Persistent per-crop extractor worker.")
    ap.add_argument("--env", required=True, choices=list(ENV_OPS),
                    help="which extractor environment this worker serves")
    a = ap.parse_args()

    real_out = sys.stdout
    devnull = open(os.devnull, "w")
    mods = {}
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for op in ENV_OPS[a.env]:
                m = importlib.import_module(op)
                m.load()
                mods[op] = m
    except Exception:
        real_out.write(ACK + json.dumps(
            {"ok": False, "ready": False, "error": traceback.format_exc()[-500:]}) + "\n")
        real_out.flush()
        return
    real_out.write(ACK + json.dumps({"ok": True, "ready": True}) + "\n")
    real_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception as exc:
            real_out.write(ACK + json.dumps({"ok": False, "op": None, "error": f"malformed job line: {exc}"}) + "\n")
            real_out.flush()
            continue
        op = job.get("op")
        if op == "__quit__":
            break
        resp = {"ok": True, "op": op}
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                mods[op].run(job.get("args"))
        except Exception:
            resp = {"ok": False, "op": op, "error": traceback.format_exc()[-500:]}
        _free_gpu()
        real_out.write(ACK + json.dumps(resp) + "\n")
        real_out.flush()


if __name__ == "__main__":
    main()
