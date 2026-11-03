#!/usr/bin/env python3
"""Crop one photograph and report its four assessments (and, with --explain, a written explanation).

    envs/main/venv/bin/python inference/run_image.py photo.jpg --out output/photo [--aspect 16:9] [--mode full] [--explain]

Writes crop.jpg and result.json (see OUTPUT_SCHEMA.md) into --out.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from photoguider import crop_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Crop one image and report calibrated composition scores.")
    ap.add_argument("image", help="input image to crop")
    ap.add_argument("--aspect", help="optional target ratio, for example 16:9")
    ap.add_argument("--mode", choices=("fast", "full"), default="fast",
                    help="crop mode (default: fast; full enables the consultant)")
    ap.add_argument("--explain", action="store_true", help="add a written explanation (GPU explainer environment required)")
    ap.add_argument("--no-strict-aspect", dest="strict_aspect", action="store_false",
                    help="with --aspect: keep the benchmark protocol's box even where a border clip changed its ratio")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    crop_image(a.image, a.aspect, a.mode, a.explain, a.out, strict_aspect=a.strict_aspect)


if __name__ == "__main__":
    main()
