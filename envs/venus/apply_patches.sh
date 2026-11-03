#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WEIGHTS_DIR=${1:-"$HERE/weights/Venus-Q-Stage2"}
TOKENIZER="$WEIGHTS_DIR/tokenization_qwen.py"
PATCH_FILE="$HERE/patches/tokenization_qwen_init_order.patch"

[[ -f $TOKENIZER ]] || { echo "Venus tokenizer not found: $TOKENIZER" >&2; exit 1; }

is_applied() {
  awk '
    /^    def __init__\(/ { in_init=1 }
    in_init && /^    def __getstate__\(/ { exit }
    in_init && /[^[:space:]]/ { last=$0 }
    END { exit(last == "        super().__init__(**kwargs)" ? 0 : 1) }
  ' "$TOKENIZER"
}

if is_applied; then
  echo "Venus tokenizer patch already applied."
  exit 0
fi

(
  cd "$WEIGHTS_DIR"
  patch -p0 --forward tokenization_qwen.py < "$PATCH_FILE"
)
is_applied || { echo "Venus tokenizer patch did not produce the expected init order." >&2; exit 1; }
echo "Applied Venus tokenizer init-order patch."
