#!/usr/bin/env bash
# Download author-distributed FLMS annotations; optionally download images.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${FLMS_CACHE_DIR:-"$ROOT/caches"}
[[ $# == 0 || ( $# == 1 && $1 == --with-images ) ]] || { echo 'Usage: download_flms_gt.sh [--with-images]' >&2; exit 2; }
mkdir -p "$DEST"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fL --retry 3 http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_gt/release_data.tar -o "$TMP/release_data.tar"
echo "754d5a9cead6cd658b0ff59866d6cd75ff304c88d88a5d245d4af9dfbedb4c2e  $TMP/release_data.tar" | sha256sum -c -
tar -xf "$TMP/release_data.tar" -C "$TMP"
mapfile -t MATS < <(find "$TMP" -type f -name 500_image_dataset.mat)
[[ ${#MATS[@]} == 1 ]] || { echo 'Expected one 500_image_dataset.mat' >&2; exit 1; }
echo "716ae0885439456fee83d92ae6940586ccc4c9d54889d1bda185b13f1f8ae6db  ${MATS[0]}" | sha256sum -c -
cp "${MATS[0]}" "$DEST/flms_500_gt.mat"
if [[ ${1:-} == --with-images ]]; then
  if [[ ! -f $DEST/flms_image.tar ]]; then
    curl -fL --retry 3 http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_image/image.tar -o "$TMP/image.tar"
    mv "$TMP/image.tar" "$DEST/flms_image.tar"
  fi
  echo "d809f4bd9a28dcaebdd4ffaa3f3d6753a914046e0ed16c3bfb78ae67b15fb9fc  $DEST/flms_image.tar" | sha256sum -c -
fi
