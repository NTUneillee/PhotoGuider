#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # envs/ -> release root
ENVS="$ROOT/envs"
CUDA_INDEX=https://download.pytorch.org/whl/cu128
CLIP_SPEC='clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'
FGAESQ_TREE="$ENVS/fgaesq/FGAesQ_Inference"
FGAESQ_WEIGHT="$ENVS/fgaesq/weights/FGAesQ.pt"
EXPLAINER_ADAPTER="$ROOT/inference/models/explainer"
WITH_WEIGHTS=0

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:3] == (3, 12, 7) else 1)'; then
  echo "Python 3.12.7 is required for the validated environment." >&2
  exit 2
fi

if [[ ${1:-} == --with-weights ]]; then
  WITH_WEIGHTS=1
  shift
fi
if (( $# )); then
  echo "Usage: bash envs/setup_envs.sh [--with-weights]" >&2
  exit 2
fi

# Validate the source revision before installing environments.
python3 "$ENVS/verify_assets.py" --revision fgaesq_source >/dev/null || exit 1

names=(main laion fgaesq sam venus explainer)
declare -A results

smoke() {
  local name=$1 python="$ENVS/$1/venv/bin/python"
  [[ -x $python ]] || return 1
  case "$name" in
    main) "$python" -c 'import torch, transformers, cv2; assert cv2.__version__ == "4.13.0", cv2.__version__' \
            >/dev/null 2>&1 ;;
    laion) "$python" -c 'import open_clip' >/dev/null 2>&1 ;;
    fgaesq)
      PYTHONPATH="$FGAESQ_TREE${PYTHONPATH:+:$PYTHONPATH}" \
        "$python" -c 'import clip; from utils.FGAesQ import FGAesQ; from utils.DiffToken import DiffToken' \
        >/dev/null 2>&1
      ;;
    sam) "$python" -c 'import segment_anything' >/dev/null 2>&1 ;;
    venus)
      "$python" -c 'import torch, transformers' >/dev/null 2>&1 || return 1
      ;;
    explainer)
      "$python" -c 'import peft, bitsandbytes, transformers' >/dev/null 2>&1 || return 1
      if [[ -d $EXPLAINER_ADAPTER ]]; then
        [[ -f $EXPLAINER_ADAPTER/adapter_model.safetensors ]] || {
          echo "Explainer adapter directory is incomplete: $EXPLAINER_ADAPTER" >&2
          return 1
        }
      else
        echo "Explainer adapter smoke skipped: adapter not downloaded at $EXPLAINER_ADAPTER"
      fi
      ;;
  esac
}

checkout_fgaesq() {
  local revision
  revision=$(python3 "$ENVS/verify_assets.py" --revision fgaesq_source) || return 1
  local checkout="$ENVS/fgaesq/FG-IAA"
  if [[ ! -d $checkout/.git ]]; then
    git clone https://github.com/yzc-ippl/FG-IAA "$checkout" || return 1
  fi
  git -C "$checkout" checkout --detach "$revision" || return 1
  [[ -d $checkout/FGAesQ_Inference ]] || {
    echo "FG-IAA checkout lacks FGAesQ_Inference: $checkout" >&2
    return 1
  }
  if [[ -e $FGAESQ_TREE && ! -L $FGAESQ_TREE ]]; then
    echo "Existing FGAesQ source is not a pinned checkout link: $FGAESQ_TREE" >&2
    return 1
  fi
  ln -sfn "$checkout/FGAesQ_Inference" "$FGAESQ_TREE" || return 1
}

install_env() {
  local name=$1 venv="$ENVS/$1/venv" pip="$ENVS/$1/venv/bin/pip"

  if [[ $name == fgaesq ]]; then
    checkout_fgaesq || return 1
  fi
  python3 -m venv "$venv" || return 1
  # Validated build tools; CLIP requires the older setuptools in fgaesq.
  local setuptools_version=81.0.0
  [[ $name == fgaesq ]] && setuptools_version=80.10.2
  "$pip" install pip==26.2.1 wheel==0.48.0 "setuptools==$setuptools_version" || return 1

  "$pip" install -r "$ENVS/$name/requirements.txt" \
    --extra-index-url "$CUDA_INDEX" || return 1

  if [[ $name == fgaesq ]]; then
    "$pip" install --no-build-isolation --no-deps "$CLIP_SPEC" || return 1
  fi
  # opencv-python and opencv-python-headless share the cv2 module and the last install wins;
  # the shipped numbers were produced with the headless build, so install it last
  if [[ $name == main ]]; then
    "$pip" install --force-reinstall --no-deps opencv-python-headless==4.13.0.92 || return 1
  fi

  smoke "$name" || return 1
  results[$name]=installed
}

overall=0
for name in "${names[@]}"; do
  echo "==> $name"
  if ! install_env "$name"; then
    results[$name]=failed
    overall=1
  fi
done

if (( WITH_WEIGHTS )); then
  mkdir -p "$ENVS/sam/weights" "$ENVS/laion/weights"
  if [[ ! -s $ENVS/sam/weights/sam_vit_b_01ec64.pth ]]; then
    curl -fL https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
      -o "$ENVS/sam/weights/sam_vit_b_01ec64.pth.part" && mv "$ENVS/sam/weights/sam_vit_b_01ec64.pth.part" "$ENVS/sam/weights/sam_vit_b_01ec64.pth" || exit 1
  fi
  if [[ ! -s $ENVS/laion/weights/sa_0_4_vit_l_14_linear.pth ]]; then
    curl -fL https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth \
      -o "$ENVS/laion/weights/sa_0_4_vit_l_14_linear.pth.part" && mv "$ENVS/laion/weights/sa_0_4_vit_l_14_linear.pth.part" "$ENVS/laion/weights/sa_0_4_vit_l_14_linear.pth" || exit 1
  fi
  # The validated build supplied a content hash, not a weight-repository commit.
  mkdir -p "$(dirname "$FGAESQ_WEIGHT")"
  if [[ -s $FGAESQ_WEIGHT ]]; then
    results[fgaesq_weight]=skipped
  elif curl -fL \
    "https://huggingface.co/yzc002/FGAesQ/resolve/main/FGAesQ.pt" \
    -o "$FGAESQ_WEIGHT.part" && mv "$FGAESQ_WEIGHT.part" "$FGAESQ_WEIGHT"; then
    results[fgaesq_weight]=downloaded
  else
    results[fgaesq_weight]=failed
    overall=1
  fi
fi

if (( WITH_WEIGHTS )); then
  python3 "$ENVS/verify_assets.py" --require-present || overall=1
fi

printf '\n%-18s %s\n' ENVIRONMENT STATUS
printf '%-18s %s\n' ------------------ ---------
for name in "${names[@]}"; do
  printf '%-18s %s\n' "$name" "${results[$name]:-failed}"
done
if (( WITH_WEIGHTS )); then
  printf '%-18s %s\n' fgaesq-weight "${results[fgaesq_weight]:-failed}"
fi
exit "$overall"
