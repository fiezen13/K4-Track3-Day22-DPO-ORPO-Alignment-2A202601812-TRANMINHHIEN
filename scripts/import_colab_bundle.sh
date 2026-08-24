#!/usr/bin/env bash
# Unpack a Colab lab22 zip into this repo.
# Usage (from repo root):
#   bash scripts/import_colab_bundle.sh ~/Downloads/lab22_core.zip
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="${1:-}"

if [[ -z "$ZIP" || ! -f "$ZIP" ]]; then
  echo "Usage: bash scripts/import_colab_bundle.sh /path/to/lab22_core.zip"
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[import] Unpacking $ZIP ..."
unzip -q "$ZIP" -d "$TMP"

# Zip may contain lab22/ or _submit/ or flat folders
SRC="$TMP"
if [[ -d "$TMP/lab22" ]]; then SRC="$TMP/lab22"; fi
if [[ -d "$TMP/_submit" ]]; then SRC="$TMP/_submit"; fi

copy_tree() {
  local rel="$1"
  if [[ -e "$SRC/$rel" ]]; then
    mkdir -p "$REPO_ROOT/$(dirname "$rel")"
    cp -a "$SRC/$rel" "$REPO_ROOT/$rel"
    echo "  ✓ $rel"
  else
    echo "  · missing in zip: $rel"
  fi
}

copy_tree "submission/screenshots"
copy_tree "data/pref"
copy_tree "data/eval"
copy_tree "adapters/sft-mini"
copy_tree "adapters/dpo"

# Optional notebook with outputs
if [[ -f "$SRC/Lab22_DPO_T4.ipynb" ]]; then
  cp -a "$SRC/Lab22_DPO_T4.ipynb" "$REPO_ROOT/colab/Lab22_DPO_T4.ipynb"
  echo "  ✓ colab/Lab22_DPO_T4.ipynb"
elif [[ -f "$SRC/colab/Lab22_DPO_T4.ipynb" ]]; then
  cp -a "$SRC/colab/Lab22_DPO_T4.ipynb" "$REPO_ROOT/colab/Lab22_DPO_T4.ipynb"
  echo "  ✓ colab/Lab22_DPO_T4.ipynb"
fi

echo
echo "[import] Done. Next:"
echo "  1. Add any screenshots you saved outside the zip into submission/screenshots/"
echo "  2. Fill submission/REFLECTION.md"
echo "  3. make verify"
echo "  4. git add / commit / push (when you ask)"
