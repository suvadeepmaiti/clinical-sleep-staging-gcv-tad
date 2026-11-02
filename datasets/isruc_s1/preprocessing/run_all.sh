#!/usr/bin/env bash
# ISRUC-Sleep subgroup I (data/raw/ISRUC-S1/<id>/<id>.rec + scorer files) -> one .npz per recording.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY preprocess_isruc.py --workers 3  # -> data/preprocessed/isruc/
$PY validate_isruc.py
