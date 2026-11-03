#!/bin/sh
# Archive: 149,549,284 bytes (~143 MB) compressed; adapter_model.safetensors:
# 161,539,072 bytes (~161 MB) extracted.
set -eu
REPO="NTUneillee/PhotoGuider"
TAG="release"
ASSET="explainer.tar.zst"
SHA256_EXPECTED="4913098dd35c520f655e146be95a519dccaee6808762ed085d45c1bc56a19595"
ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/inference/models"
TARGET="$DEST/explainer"
MANIFEST=".extracted.sha256"
FORCE=0
usage() { echo "Usage: $0 [--force]"; }
case ${1-} in "") ;; --force) FORCE=1 ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
mkdir -p "$DEST"
if [ "$FORCE" -eq 1 ] && [ -d "$TARGET" ]; then rm -rf "$TARGET"; fi
if [ -d "$TARGET" ] && [ -f "$TARGET/$MANIFEST" ]; then
  if (cd "$TARGET" && sha256sum -c "$MANIFEST" >/dev/null); then
    echo "Verified explainer/ against its extracted-file digest manifest."; exit 0
  fi
  echo "Existing explainer/ failed extracted-file verification; downloading a clean copy." >&2
  rm -rf "$TARGET"
elif [ -d "$TARGET" ]; then
  echo "Existing explainer/ has no extracted-file digest manifest; downloading a clean copy." >&2
  rm -rf "$TARGET"
fi
cd "$DEST"
echo "Downloading ${ASSET} from GitHub ${REPO}@${TAG} ..."
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --clobber
else
  curl -fL -o "$ASSET" "https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
fi
echo "Verifying archive sha256 ..."
echo "${SHA256_EXPECTED}  ${ASSET}" | sha256sum -c -
echo "Extracting ..."
zstd -d < "$ASSET" | tar -xf -
rm -f "$ASSET"
if [ ! -f "$TARGET/adapter_model.safetensors" ]; then echo "Archive did not create the expected adapter." >&2; exit 1; fi
(cd "$TARGET" && find . -type f ! -name "$MANIFEST" -print | LC_ALL=C sort | xargs sha256sum > "$MANIFEST")
(cd "$TARGET" && sha256sum -c "$MANIFEST" >/dev/null)
echo "Done: adapter in $TARGET; extracted digests recorded."
