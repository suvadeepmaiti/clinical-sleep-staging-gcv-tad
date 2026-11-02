#!/usr/bin/env bash
# Sleep-EDF-20: download the 39 SC nights from PhysioNet (SHA-256 checked), make one .npz per night, run the tests.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY download_sleepedf20.py      # -> data/raw/sleep-cassette/
$PY preprocess_sleepedf20.py    # -> data/preprocessed/sleepedf20/
$PY -m unittest test_preprocessing
