#!/usr/bin/env bash
# Analyses and paper outputs, after every cohort's modelling/run_all.sh. Writes <results>/analysis/ and paper/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
$PY fragmentation.py            # stage changes per hour, N1 share, per cohort
$PY benchmark_efficiency.py     # parameters, MACs, size, CPU latency (paper/tables/efficiency.tex)
$PY deck_data.py                # pooled confusion matrices and the example patient
$PY xai_calibration.py          # hypnodensity + reliability (calibration.json)
$PY -u distillation_map.py      # where the G_cv-TAD loss acts (GPU inference with the saved checkpoints)
$PY plot_teacher_vs_gain.py     # per-patient gain vs teacher accuracy
cd ../paper
for s in make_tables.py fig_pipeline.py fig_mask.py fig_confusion.py fig_distill_map.py fig_hypnodensity.py; do $PY "$s"; done
