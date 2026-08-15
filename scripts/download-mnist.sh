#!/usr/bin/env bash
#
# Downloads the MNIST dataset into src/data/.
#
# Sources: the ossci-datasets S3 mirror.
#
# Usage:
#   ./scripts/download-mnist.sh          # download anything missing
#   ./scripts/download-mnist.sh --force  # re-download everything

set -euo pipefail

BASE_URLS=(
  "https://ossci-datasets.s3.amazonaws.com/mnist"
)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$(dirname "$SCRIPT_DIR")/src/data"

# gz name -> decompressed name (as expected by the examples)
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
  fetched=0
  for base in "${BASE_URLS[@]}"; do
    echo "fetch $base/$gz"
    if curl -fsSL --max-time 120 "$base/$gz" -o "$tmp"; then
      fetched=1
      break
    fi
    echo "  failed, trying next source..."
  done

  if [[ $fetched -eq 0 ]]; then
    rm -f "$tmp"
    echo "error: could not download $gz from any source" >&2
    exit 1
  fi

  gunzip -c "$tmp" > "$DEST_DIR/$out"
  rm -f "$tmp"
  echo "write $out"
done

echo "Done. MNIST data is ready in src/data/"
