#!/usr/bin/env bash
#
# Downloads the MNIST dataset into src/data/, and mirrors it into dist/data/
# when a build exists. Usage: ./scripts/download-mnist.sh [--force]

set -euo pipefail

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO_DIR/src/data"

FILES=(
  "train-images-idx3-ubyte.gz:train-images.idx3-ubyte"
  "train-labels-idx1-ubyte.gz:train-labels.idx1-ubyte"
  "t10k-images-idx3-ubyte.gz:t10k-images.idx3-ubyte"
  "t10k-labels-idx1-ubyte.gz:t10k-labels.idx1-ubyte"
)

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "$DEST_DIR"

for entry in "${FILES[@]}"; do
  gz="${entry%%:*}"
  out="${entry##*:}"

  if [[ -f "$DEST_DIR/$out" && $FORCE -eq 0 ]]; then
    echo "skip  $out (already present; use --force to re-download)"
    continue
  fi

  tmp="$(mktemp "$DEST_DIR/.mnist.XXXXXX")"
  curl -fsSL --max-time 120 "$BASE_URL/$gz" -o "$tmp" || {
    rm -f "$tmp"
    echo "error: could not download $gz" >&2
    exit 1
  }
  gunzip -c "$tmp" > "$DEST_DIR/$out"
  rm -f "$tmp"
  echo "write $out"
done

if [[ -d "$REPO_DIR/dist" ]]; then
  mkdir -p "$REPO_DIR/dist/data"
  cp "$DEST_DIR"/*.idx[13]-ubyte "$REPO_DIR/dist/data/"
fi

echo "Done. MNIST data is ready in src/data/ (mirrored to dist/data/ if dist exists)"
