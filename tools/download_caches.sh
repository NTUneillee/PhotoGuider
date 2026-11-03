#!/bin/sh
# Download and extract caches required by reproduction, training, and single-image inference.
set -eu
REPO="NTUneillee/PhotoGuider"
TAG="release"
ASSET="caches.tar.zst"
SHA256_EXPECTED="902e38f8d5739fcadfce89cee917f33854fcc7811ad2443f47d6425a2528d53e"
ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
MANIFEST=".extracted.sha256"
FORCE=0
usage() { echo "Usage: $0 [--force]"; }
case ${1-} in "") ;; --force) FORCE=1 ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
cd "$ROOT"
if [ "$FORCE" -eq 1 ] && [ -d "$ROOT/caches" ]; then rm -rf "$ROOT/caches"; fi
if [ -d caches ] && [ -f "caches/$MANIFEST" ]; then
  if (cd caches && sha256sum -c "$MANIFEST" >/dev/null); then
    echo "Verified caches/ against its extracted-file digest manifest."; exit 0
  fi
  echo "Existing caches/ failed extracted-file verification; downloading a clean copy." >&2
  rm -rf "$ROOT/caches"
elif [ -d caches ]; then
  echo "Existing caches/ has no extracted-file digest manifest; downloading a clean copy." >&2
  rm -rf "$ROOT/caches"
fi
echo "Downloading ${ASSET} from ${REPO}@${TAG} ..."
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
if [ ! -d caches ]; then echo "Archive did not create the expected caches/ directory." >&2; exit 1; fi
(cd caches && find . -type f ! -name "$MANIFEST" -print | LC_ALL=C sort | xargs sha256sum > "$MANIFEST")
(cd caches && sha256sum -c "$MANIFEST" >/dev/null)
echo "Done: $(find caches -type f ! -name "$MANIFEST" | wc -l | tr -d ' ') files in caches/; extracted digests recorded."
