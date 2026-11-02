#!/usr/bin/env bash
# ISRUC-S1 (sleep-disorder clinic, 99 of 100 recordings): the runs reported in the paper, in order.
# 1) 10-fold patient-wise CV with all labels: masked SleepGPT-BiLSTM teacher + causal CE students
#    (+ balanced accuracy / weighted F1 for the comparison with published results).
# 2) 25% of the training patients labelled: CE, KD and G_cv-TAD students; paired per-patient tests.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
PY=${PY:-python}
R=${SLEEP_RESULTS:-$REPO/results}/isruc
D=${SLEEP_DATA:-$REPO/data}/preprocessed/isruc

$PY -u isruc_cv.py --folds 1 2 3 4 5 6 7 8 9 10 --student-variants ce_only --output "$R/isruc_cv10_ce"
$PY extra_metrics.py --results "$R/isruc_cv10_ce"
$PY -u "$REPO/src/label_efficiency.py" --dataset isruc --fraction 0.25 --folds 1 2 3 4 5 6 7 8 9 10 \
    --inputs eeg eeg_eog eog --variants ce kd gtad_cv --output "$R/label_efficiency_25pct"
$PY "$REPO/analysis/label_efficiency_stats.py" --results "$R/label_efficiency_25pct" --npz "$D" --prefix "" \
    --full "$R/isruc_cv10_ce" --layout isruc
