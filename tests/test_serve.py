"""The resident server answers bad inputs with an error ACK instead of exiting (no models involved)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
import serve  # noqa: E402


def test_missing_image_is_reported_not_fatal(tmp_path):
    resp = serve.handle({"image": str(tmp_path / "missing.jpg"), "aspect": None, "mode": "fast", "explain": False,
                         "out": str(tmp_path / "out")}, session=None)
    assert resp["ok"] is False and "not found" in resp["error"]


def test_bad_aspect_is_reported_not_fatal(tmp_path):
    resp = serve.handle({"image": str(tmp_path / "missing.jpg"), "aspect": "sixteen:nine", "mode": "fast",
                         "explain": False, "out": str(tmp_path / "out")}, session=None)
    assert resp["ok"] is False and "aspect" in resp["error"]
    json.dumps(resp)                                   # serialisable as an ACK line
