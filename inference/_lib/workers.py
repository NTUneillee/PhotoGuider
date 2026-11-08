"""Persistent extractor worker pool.

Each per-crop feature extractor runs in its own Python environment and, cold,
reloads a heavy model (CLIP-L, LAION ViT-L, FGAesQ, SAM) on every invocation.
This pool starts one long-lived worker process per environment, each of which
loads its model once and then serves extraction jobs over a line-based JSON
protocol, so repeated images pay the model-load cost only once.

Usage:
    pool = ExtractorPool()
    ex = Extractors(scratch, runner=pool.run_op)      # feature_extractors.Extractors served warm
    ...
    pool.close()

``run_op(op, args)`` is the runner signature feature_extractors.Extractors takes; the op's
environment comes from the shared OP_ENV registry. If a worker fails to start (missing
environment), that environment transparently falls back to a one-shot subprocess.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from env_config import env_python
sys.path.insert(0, str(HERE))
from feature_extractors import OP_ENV, run_cold
EXTRACTORS = HERE / "extractors"
WORKER = EXTRACTORS / "worker_serve.py"
ACK = "@@ACK@@"

ENV_PY = {name: env_python(name) for name in ("main", "laion", "fgaesq", "sam")}


class ExtractorPool:
    def __init__(self, verbose=False):
        self.procs = {}          # env_key -> Popen (or "cold" sentinel if unavailable)
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[pool] {msg}", flush=True)

    def warm(self):
        """Start every extractor worker now instead of on the first request."""
        for env_key in ENV_PY:
            self._ensure(env_key)

    def _ensure(self, env_key):
        """Return a live worker for env_key, or None if it must fall back to cold runs."""
        p = self.procs.get(env_key)
        if p == "cold":
            return None
        if p is not None and p.poll() is None:
            return p
        py = ENV_PY[env_key]
        if not Path(py).exists():
            self._log(f"env '{env_key}' interpreter missing -> cold fallback")
            self.procs[env_key] = "cold"
            return None
        self._log(f"starting '{env_key}' worker ...")
        proc = subprocess.Popen(
            [str(py), "-u", str(WORKER), "--env", env_key],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)   # stray lines are forwarded by _read_ack
        resp = self._read_ack(proc)
        if resp is None or not resp.get("ready"):
            self._log(f"env '{env_key}' worker failed to load -> cold fallback")
            try:
                proc.kill()
            except Exception:
                pass
            self.procs[env_key] = "cold"
            return None
        self.procs[env_key] = proc
        return proc

    def _read_ack(self, proc):
        """Read stdout lines until an ACK marker; forward stray lines to stderr."""
        while True:
            line = proc.stdout.readline()
            if not line:
                return None
            if line.startswith(ACK):
                try:
                    return json.loads(line[len(ACK):])
                except Exception:
                    return None
            else:
                sys.stderr.write(f"[worker] {line}")

    def run_op(self, op, args=None):
        """Run one extraction op (script stem) with ``args`` (must carry "scratch"). True on success."""
        env_key = OP_ENV.get(op)
        if env_key is None:
            raise ValueError(f"unknown extractor op '{op}'")
        proc = self._ensure(env_key)
        if proc is None:
            return run_cold(op, args)
        try:
            proc.stdin.write(json.dumps({"op": op, "args": args}) + "\n")
            proc.stdin.flush()
        except Exception:
            self.procs[env_key] = None
            return run_cold(op, args)
        resp = self._read_ack(proc)
        if resp is None:
            self._log(f"worker '{env_key}' died on '{op}' -> cold retry")
            self.procs[env_key] = None
            return run_cold(op, args)
        if not resp.get("ok"):
            # extractor failures must be visible even without verbose mode: this is the only
            # place the worker's traceback surfaces (its stdout/stderr go to devnull)
            print(f"[pool] op '{op}' FAILED in '{env_key}' worker:\n{resp.get('error')}",
                  file=sys.stderr, flush=True)
            return False
        return True

    def close(self):
        for env_key, proc in list(self.procs.items()):
            if proc == "cold" or proc is None:
                continue
            try:
                proc.stdin.write(json.dumps({"op": "__quit__"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.procs.clear()
