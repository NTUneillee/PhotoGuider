#!/usr/bin/env python3
"""Resident PhotoGuider server for the web app: one Session, many requests.

    envs/main/venv/bin/python inference/serve.py

Reads one JSON request per line on stdin and runs it with a Session kept across requests, so the
extractor workers load their models once (the explainer still runs per request). Pipeline output
goes to stdout as it is produced; each request ends with one line
``@@ACK@@{"ok": true, "result": <result.json>}`` or ``@@ACK@@{"ok": false, "error": <text>}``.

    {"image": "photo.jpg", "aspect": "16:9" | null, "mode": "fast", "explain": true, "out": "output/photo"}
    {"pano": "panorama.jpg", "mode": "fast", "explain": true, "hdr": "auto", "out": "output/pano"}
"""
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "PanoCropping" / "scripts")]
from photoguider import crop_image, Session  # noqa: E402
import run_pano  # noqa: E402

ACK = "@@ACK@@"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    session = Session(verbose=True)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                if "image" in req:
                    crop_image(req["image"], req.get("aspect"), req["mode"], req["explain"], req["out"], session=session)
                else:
                    run_pano.run(req["pano"], req["mode"], req["explain"], req["out"], hdr_path=req.get("hdr"),
                                 session=session)
                resp = {"ok": True, "result": str(Path(req["out"]).resolve() / "result.json")}
            except Exception as exc:                 # SystemExit included: bad inputs are reported this way
                resp = {"ok": False, "error": str(exc) or traceback.format_exc()[-1500:]}
            print(ACK + json.dumps(resp), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
