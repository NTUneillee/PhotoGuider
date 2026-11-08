"""Shared, lazily imported implementation of the explanation VLM."""
import json
import re
from pathlib import Path
from model_pins import revision

BASE_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 160
# pixel cap for the crop the VLM sees: the 60-degree panorama working view (1600 x 1067), the largest
# input the released explanations were produced on; larger camera crops are downscaled to it so the
# visual tokens fit on a 16 GB GPU (smaller crops are unaffected)
MAX_PIXELS = 1600 * 1067
CJK_RX = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")


def cjk_bad_words_ids(tok, cache_path):
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        if data.get("n_vocab") == len(tok):
            return [[i] for i in data["ids"]]
    bad = [i for i in range(len(tok)) if CJK_RX.search(tok.decode([i]))]
    cache_path.write_text(json.dumps({"n_vocab": len(tok), "ids": bad}))
    return [[i] for i in bad]


def load_vlm(adapter_dir):
    """Load NF4 Qwen + PEFT adapter. Heavy dependencies stay lazy."""
    adapter_dir = Path(adapter_dir)
    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    from peft import PeftModel
    assert (adapter_dir / "adapter_model.safetensors").exists(), (
        f"explainer adapter not found in {adapter_dir} — run tools/download_explainer.sh")
    torch.manual_seed(1234)
    nf4 = adapter_dir.parent / "explainer_nf4"
    if (nf4 / "config.json").exists():
        # the same NF4 weights saved once (tools/save_explainer_nf4.py): 6 GB to read instead of the
        # 16 GB bf16 checkpoint quantized on every load; the decoded text is identical
        proc = AutoProcessor.from_pretrained(nf4)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(nf4, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 llm_int8_skip_modules=["visual", "lm_head"])
        proc = AutoProcessor.from_pretrained(BASE_VLM, revision=revision("qwen"))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_VLM, revision=revision("qwen"), device_map="auto", quantization_config=bnb,
            torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir),
                                      is_trainable=False).eval()
    return {"proc": proc, "model": model,
            "bad": cjk_bad_words_ids(proc.tokenizer,
                                     adapter_dir / "cjk_ids_cache.json")}


def generate_explanation(crop_path, prompt, adapter_dir):
    """Single greedy decode with the release CJK token mask."""
    import torch
    state = load_vlm(adapter_dir)
    proc, model, bad = state["proc"], state["model"], state["bad"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(crop_path), "max_pixels": MAX_PIXELS},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids,
                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=MAX_NEW_TOKENS, bad_words_ids=bad)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)[0].strip()
