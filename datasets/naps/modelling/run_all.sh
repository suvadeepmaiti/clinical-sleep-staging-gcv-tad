#!/usr/bin/env bash
# NAPS (isolated REM sleep behaviour disorder, 133 patients): the runs reported in the paper, in order.
# 1) 10-fold patient-wise CV with all labels: masked SleepGPT-BiLSTM teacher + causal CE students.
# 2) 25% of the training patients labelled: CE, KD and G_cv-TAD students; paired per-patient tests.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
PY=${PY:-python}
R=${SLEEP_RESULTS:-$REPO/results}/naps
D=${SLEEP_DATA:-$REPO/data}/preprocessed/naps

$PY -u naps_cv.py --folds 1 2 3 4 5 6 7 8 9 10 --student-variants ce_only --output "$R/naps_cv10_ce"
$PY -u "$REPO/src/label_efficiency.py" --dataset naps --fraction 0.25 --folds 1 2 3 4 5 6 7 8 9 10 \
    --inputs eeg eeg_eog eog --variants ce kd gtad_cv --output "$R/label_efficiency_25pct"
$PY "$REPO/analysis/label_efficiency_stats.py" --results "$R/label_efficiency_25pct" --npz "$D" --prefix "" \
    --full "$R/naps_cv10_ce" --layout naps
