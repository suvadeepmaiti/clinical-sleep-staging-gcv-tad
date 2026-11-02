#!/usr/bin/env bash
# iSLEEPS (ischaemic stroke, 99 patients): the runs reported in the paper, in order.
# 1) 10-fold patient-wise CV with all labels: SleepGPT-BiLSTM teacher (one seed) + causal CE students.
# 2) 25% of the training patients labelled: CE, KD and G_cv-TAD students; paired per-patient tests.
# Needs data/cache/isleeps/psg4_100hz (datasets/isleeps/preprocessing) and the SleepGPT checkpoint (third_party/).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
PY=${PY:-python}
R=${SLEEP_RESULTS:-$REPO/results}/isleeps
D=${SLEEP_DATA:-$REPO/data}/preprocessed/isleeps

for f in 1 2 3 4 5 6 7 8 9 10; do
  $PY -u run_sleepgpt4_cv.py --seeds 1 --folds "$f" --output "$R/sleepgpt4_cv10_s1"
  $PY -u "$REPO/src/train_students_context.py" --folds "$f" --source "$R/sleepgpt4_cv10_s1" \
      --inputs eog c4 eeg_eog --variants ce_only --skip-perepoch --output "$R/context_students_cv10_s1"
done
$PY aggregate_context.py --results "$R/context_students_cv10_s1" --teacher "$R/sleepgpt4_cv10_s1"

$PY -u "$REPO/src/label_efficiency.py" --dataset isleeps --fraction 0.25 --folds 1 2 3 4 5 6 7 8 9 10 \
    --inputs eeg eeg_eog eog --variants ce kd gtad_cv --output "$R/label_efficiency_25pct"
$PY "$REPO/analysis/label_efficiency_stats.py" --results "$R/label_efficiency_25pct" --npz "$D" --prefix SN \
    --full "$R/context_students_cv10_s1" --layout isleeps
