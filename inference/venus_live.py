"""Single-image live decode of the consultant MLLM (Venus) — full-mode in-the-wild inference.

Adapts the consultant artifact-generation runs EXACTLY (the same prompts, seeds and parsing that
produced the cached benchmark boxes):
  - greedy   : 1 box, greedy decode
  - multibox : 5 boxes, temperature samples (top-p 0.9, T=1.0), seeds 1000..1004
  - promptens: 4 boxes, greedy, one per paraphrased prompt

decode_image(image_path) -> {"greedy":[x1,y1,x2,y2], "multibox":[..5..], "promptens":[..4..]}

Model loaded once per process (lazy singleton). 4-bit NF4 quant, deterministic
(seed reset before each decode). Robust parse: malformed output -> retry once,
then raise a clear error (greedy) or keep a null slot (multibox / promptens).

CLI (what infer.consult runs):
  python venus_live.py --image PATH --out out.json
"""
import os, re, json, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from env_config import asset

MP = str(asset("venus_model", "envs/venus/weights/Venus-Q-Stage2"))
VENUS_PATCHER = "bash envs/venus/apply_patches.sh"

# ── canonical prompts (must match the artifact-generation runs EXACTLY) ────────
PROMPT_MAIN = ('Please provide the bounding box coordinate of the most visually '
               'balanced and aesthetically pleasing composition area.')  # greedy + multibox
PROMPTENS = [  # 4 paraphrases, greedy each
    "Please provide the bounding box coordinate of the best aesthetic crop of this image.",
    "Please provide the bounding box coordinate of the most well-composed cropping region.",
    "Please provide the bounding box coordinate of the ideal crop that improves the composition of this photo.",
    "Please provide the bounding box coordinate of the most visually pleasing crop region following photographic composition principles.",
]
K_MULTI = 5  # temperature samples, seeds 1000..1004

_tok = None
_model = None


def _load():
    """Load Venus-Q-Stage2 4-bit once per process."""
    global _tok, _model
    if _model is not None:
        return
    tokenizer_code = Path(MP) / "tokenization_qwen.py"
    try:
        init_body = tokenizer_code.read_text(encoding="utf-8").split("    def __getstate__(", 1)[0]
        last_statement = next(line.strip() for line in reversed(init_body.splitlines()) if line.strip())
    except (OSError, StopIteration):
        last_statement = ""
    if last_statement != "super().__init__(**kwargs)":
        raise SystemExit(f"Venus tokenizer patch missing; run `{VENUS_PATCHER}`.")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # The upstream tokenizer module downloads SimSun.ttf into the *current directory* on first
    # import. Load it with the weights directory as cwd so the font lands next to the weights
    # (downloaded once) instead of polluting whatever directory the user ran the command from.
    _prev_cwd = os.getcwd(); os.chdir(MP)
    try:
        _tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    finally:
        os.chdir(_prev_cwd)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             llm_int8_skip_modules=["visual"])
    _model = AutoModelForCausalLM.from_pretrained(
        MP, device_map="auto", quantization_config=bnb, trust_remote_code=True).eval()


def _parse(resp, W, H):
    """Canonical parse: first two (x,y) pairs, scaled from /1000 to pixels. None if malformed."""
    pts = re.findall(r"\((\d+),(\d+)\)", resp)
    if len(pts) >= 2:
        x1, y1 = int(pts[0][0]), int(pts[0][1])
        x2, y2 = int(pts[1][0]), int(pts[1][1])
        return [x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000]
    return None


def _chat(query, do_sample, seed):
    import torch
    torch.manual_seed(seed)
    if do_sample:
        resp, _ = _model.chat(_tok, query=query, history=None,
                              do_sample=True, top_p=0.9, temperature=1.0)
    else:
        resp, _ = _model.chat(_tok, query=query, history=None)
    return resp


def _box(path, W, H, prompt, do_sample, seed, retry_seed, strict):
    """One decoded box: decode with ``seed``; on malformed output decode once more with ``retry_seed``.
    A double failure raises when ``strict`` (the greedy box) and otherwise yields None (the canonical
    artifacts keep null multibox / promptens slots; the infer.py consumer filters them)."""
    q = _tok.from_list_format([{'image': path}, {'text': prompt}])
    resp = _chat(q, do_sample, seed)
    b = _parse(resp, W, H)
    if b is None:
        resp = _chat(q, do_sample, retry_seed)
        b = _parse(resp, W, H)
    if b is None:
        if strict:
            raise ValueError(f"malformed Venus output for {path!r} prompt={prompt!r}: {resp!r}")
        print(f"[venus_live] WARN malformed output for {path!r} prompt={prompt!r}: {resp!r} -> null slot", flush=True)
    return b


def decode_image(image_path):
    """Return {'greedy':[..], 'multibox':[..5..], 'promptens':[..4..]} for one image."""
    from PIL import Image
    _load()
    path = str(image_path)
    W, H = Image.open(path).size
    # greedy (seed 1234, retry 5678), multibox (temperature samples, seeds 1000+k, retry 2000+k),
    # promptens (4 paraphrases, greedy, seeds 1234 / 5678 each, null-tolerant)
    greedy = _box(path, W, H, PROMPT_MAIN, False, 1234, 5678, strict=True)
    multibox = [_box(path, W, H, PROMPT_MAIN, True, 1000 + k, 2000 + k, strict=False) for k in range(K_MULTI)]
    promptens = [_box(path, W, H, p, False, 1234, 5678, strict=False) for p in PROMPTENS]
    return {"greedy": greedy, "multibox": multibox, "promptens": promptens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {a.image: decode_image(a.image)}
    open(a.out, "w").write(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
