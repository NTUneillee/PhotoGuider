"""The web app reports a pipeline server that dies while loading instead of crashing on the next request
(no Gradio, no GPU: Gradio is stubbed and the server is a child process that exits before READY)."""
import contextlib
import importlib.util
import io
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("gradio") is None:        # CI has only the main requirements
    gr = types.ModuleType("gradio")

    class Error(Exception):
        def __init__(self, message, duration=None):
            super().__init__(message)
    gr.Error, gr.Warning = Error, print
    sys.modules["gradio"] = gr
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

FAILING_SERVER = [sys.executable, "-c", "print('loading extractor'); print('RuntimeError: no kernel image'); raise SystemExit(3)"]
READY_SERVER = [sys.executable, "-c", "import sys; print('@@READY@@', flush=True); sys.stdin.read()"]


REAL_POPEN = subprocess.Popen


def _popen_with(argv):
    """A Popen that runs `argv` instead of the main environment's serve.py, keeping the pipe setup."""
    return lambda *_a, **kw: REAL_POPEN(argv, **{k: v for k, v in kw.items() if k != "cwd"})


def test_server_dying_before_ready_is_reported():
    app.subprocess.Popen = _popen_with(FAILING_SERVER)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert app.start_server() is False
    assert app._server is None and "no kernel image" in app._start_error
    assert "exited with code 3" in out.getvalue()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            list(app.run_streaming({"image": "x.jpg"}))
    except app.gr.Error as exc:
        assert "could not start" in str(exc) and "no kernel image" in str(exc)
    else:
        raise AssertionError("run_streaming must raise gr.Error when the server cannot start")


def test_ready_server_is_kept():
    app.subprocess.Popen = _popen_with(READY_SERVER)
    assert app.start_server() is True
    assert app._server is not None and app._start_error is None and app._server.poll() is None
    app._server.kill()
    app._server.wait()
    app._server = None


if __name__ == "__main__":
    try:
        test_server_dying_before_ready_is_reported()
        test_ready_server_is_kept()
    finally:
        app.subprocess.Popen = REAL_POPEN
    print("ok")
