#!/usr/bin/env python3
"""Download the explainer's NF4-quantized base model (6 GB) into inference/models/explainer_nf4/,
so explanations load it instead of quantizing the 16 GB bf16 checkpoint on every run. The weights
are the ones tools/save_explainer_nf4.py produces; the decoded text is identical either way.

    envs/explainer/venv/bin/python tools/download_explainer_nf4.py
"""
import sys
from pathlib import Path

REPO = "NeilLeeNTU/PhotoGuider-explainer-nf4"
REVISION = "7bf6efabd259f726325be2050096a6e7eb138342"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "inference" / "models" / "explainer_nf4"

from huggingface_hub import snapshot_download  # noqa: E402

path = snapshot_download(REPO, revision=REVISION, local_dir=DEST)
if not (DEST / "config.json").exists() or not (DEST / "model.safetensors").exists():
    sys.exit(f"download incomplete: {DEST}")
print(f"explainer NF4 base ready in {path}")
