#!/usr/bin/env bash
# iSLEEPS: raw EDF + hypnograms (data/raw/iSLEEPS) -> one .npz per patient, checks, and the 100 Hz teacher input cache.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY preprocess_isleeps.py       # -> data/preprocessed/isleeps/
$PY validate_isleeps.py
$PY check_alignment.py
$PY make_c4_cache.py            # -> data/cache/isleeps/c4_100hz/
$PY make_psg4_cache.py          # -> data/cache/isleeps/psg4_100hz/ (C3, C4, EMG, E1 at 100 Hz)
