#!/usr/bin/env bash
# NAPS (NSRR download in data/raw/naps) -> one .npz per recording. alignment_all.csv (timing diagnostics) is
# included; regenerate it with: python check_alignment_naps.py --all --harmonized-only --csv alignment_all.csv
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY preprocess_naps.py --workers 2   # -> data/preprocessed/naps/
$PY validate_naps.py
