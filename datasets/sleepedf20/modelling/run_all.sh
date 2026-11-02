#!/usr/bin/env bash
# Sleep-EDF-20 (healthy reference, 20 subjects / 39 nights): leave-one-subject-out, 20 folds.
# 1) sleepedf_cv.py: per fold, 2-channel teacher + CE student (100% labels) and 25%-label teacher + G_cv-TAD student.
# 2) sleepedf_teacher4.py: the same folds with the teacher given all four slots (chin EMG is a 1 Hz envelope
#    in these recordings, interpolated to 100 Hz). Needs the raw EDFs (data/raw/sleep-cassette).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
PY=${PY:-python}
R=${SLEEP_RESULTS:-$REPO/results}/sleepedf20

$PY -u sleepedf_cv.py --output "$R/sleepedf20_loso"
$PY -u sleepedf_teacher4.py --output "$R/sleepedf20_loso_teacher4"
