#!/usr/bin/env python3
"""One-shot JSON/stdin bridge into the explainer environment."""
import json
import sys


def main():
    request = json.load(sys.stdin)
    # Keep torch/transformers/PEFT out of module import in the system interpreter.
    from explain_vlm import generate_explanation
    explanation = generate_explanation(request["image"], request["prompt"], request["adapter_dir"])
    json.dump({"explanation": explanation}, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
