#!/usr/bin/env python3
"""Save the explainer's NF4-quantized base model once, so later loads read 6 GB instead of quantizing
the 16 GB bf16 checkpoint every time (inference/models/explainer_nf4/; the decoded text is identical).

    envs/explainer/venv/bin/python tools/save_explainer_nf4.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference" / "_lib"), str(ROOT / "inference")]
import torch  # noqa: E402
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402
import explain_vlm as EV  # noqa: E402

out = ROOT / "inference" / "models" / "explainer_nf4"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                         llm_int8_skip_modules=["visual", "lm_head"])
proc = AutoProcessor.from_pretrained(EV.BASE_VLM, revision=EV.revision("qwen"))
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(EV.BASE_VLM, revision=EV.revision("qwen"), device_map="auto",
                                                           quantization_config=bnb, torch_dtype=torch.bfloat16)
model.save_pretrained(out, safe_serialization=True)
proc.save_pretrained(out)
print(f"saved {out}")
