#!/usr/bin/env bash
# Fetch the Lenta Uplift dataset into data/raw/lenta/.
# Public mirror maintained by the scikit-uplift project (see docs/dataset_guide.md §4).
# Idempotent: skips the download if the file is already present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/lenta"
URL="https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz"

mkdir -p "$DEST"
if [[ -f "$DEST/lenta_dataset.csv.gz" ]]; then
  echo "[fetch_lenta] already present: $DEST/lenta_dataset.csv.gz"
  exit 0
fi
echo "[fetch_lenta] downloading (~145 MB) ..."
curl -fSL --retry 3 -o "$DEST/lenta_dataset.csv.gz" "$URL"
echo "[fetch_lenta] done -> $DEST/lenta_dataset.csv.gz"
