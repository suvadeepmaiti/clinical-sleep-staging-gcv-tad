# G_cv-TAD: transition-aware distillation for wearable sleep staging in patients

Code for *Towards wearable-scale, transition-aware sleep staging for clinical populations*
(S. Maiti, K. Sinha, Bapi Raju S., ISBI 2027 submission).

A large teacher (SleepGPT encoder + BiLSTM, 136.2M parameters, 4 channels, ±10 epochs of context) is fine-tuned on
the labelled patients of a cohort and distilled into a causal CNN-GRU student (≈31k parameters, 1–2 channels, one
30-s epoch at a time) that can run on a wearable. **G_cv-TAD** weights the distillation loss on consecutive epochs by
the teacher's confidence and by a disease-aware stage-transition prior

  G_cv = (1 − M̄) / (1 + λ·CV),  λ = 0.5,

estimated from the cohort's labelled patients (M̄: mean per-patient transition matrix, CV: its coefficient of
variation across patients), so the student is pushed hardest on stage changes that are typical for the cohort.
Student loss: 0.3·CE + 0.2·L_Gcv, temperature T = 2.

## Cohorts

| Key | Dataset | Population | Protocol |
|---|---|---|---|
| `isleeps` | iSLEEPS | ischaemic stroke, 99 patients | patient-wise 10-fold CV |
| `naps` | NAPS (NSRR) | isolated REM sleep behaviour disorder, 133 patients | patient-wise 10-fold CV |
| `isruc` | ISRUC-Sleep subgroup I | sleep-disorder clinic, 99 recordings | patient-wise 10-fold CV |
| `sleepedf20` | Sleep-EDF-20 (SC) | healthy reference, 20 subjects / 39 nights | leave-one-subject-out |

Each fold holds out test patients, keeps 9–10% of the rest for inner validation, and trains on the remainder. In the
label-efficiency setting only 25% of the training patients are labelled (stratified, fixed seed); the other training
patients' scorer labels are never used. All model selection uses inner-validation macro F1; test patients never enter
training, selection or the transition prior. The exact folds and labelled subsets of the published runs are in
`splits/` (the code regenerates them and, for iSLEEPS, checks them against `splits/isleeps_folds.json`).

## Repository layout

```
src/                      shared library (one flat import root)
  repo_paths.py           where data, results and the SleepGPT checkpoint live
  sleepgpt_c4.py          SleepGPT encoder re-implemented in PyTorch (loads the released checkpoint)
  teachers_multi.py       4-slot SleepGPT-BiLSTM teacher (missing channels masked)
  engine_multi.py         teacher training / prediction (exact resume)
  models.py               SleepStudentNet: the causal CNN-GRU student
  train_students_context.py   student training on runs of consecutive epochs, causal streaming inference
  train_students_gtad.py  G / G_cv transition priors and the G_cv-TAD loss
  label_efficiency.py     25%-label experiment (teacher on labelled patients; CE, KD, G_cv-TAD students)
  protocol*.py, engine.py, report*.py, run_loso.py, audit_*.py   metrics, folds, I/O, checks
datasets/<cohort>/
  preprocessing/          raw recordings -> one .npz of 30-s epochs per recording, validation, notes
  modelling/              cohort pipeline (+ run_all.sh with the exact commands of the paper)
analysis/                 fragmentation, efficiency benchmark, calibration/hypnodensity, distillation map, stats
paper/                    tables (make_tables.py) and figures (fig_*.py) of the paper; literature values
splits/                   folds and labelled subsets used for the published results
tests/                    checks of the SleepGPT re-implementation against the original code
third_party/              how to obtain SleepGPT (not redistributed)
```

## Setup

```bash
python -m pip install -r requirements.txt     # Python 3.11; a CUDA GPU is needed for training
```

Data and results are kept outside the code tree: by default `data/` and `results/` in the repository root (both
git-ignored). Set `SLEEP_DATA`, `SLEEP_RESULTS` and `SLEEPGPT_DIR` to put them elsewhere. Expected raw-data layout:

```
data/raw/sleep-cassette/   Sleep-EDF Expanded, SC subset (downloaded by the script)
data/raw/iSLEEPS/          iSLEEPS (Maiti et al., Scientific Data 2026), obtained from its authors
data/raw/naps/             NAPS from the National Sleep Research Resource (requires an NSRR data request)
data/raw/ISRUC-S1/         ISRUC-Sleep subgroup I (<id>/<id>.rec and scorer files)
```

The teacher needs the SleepGPT checkpoint; see `third_party/README.md`.

## Running

Run each step from any directory; every `run_all.sh` lists the exact commands and arguments used for the paper.

```bash
# 1. preprocessing (per cohort)
datasets/sleepedf20/preprocessing/run_all.sh
datasets/isleeps/preprocessing/run_all.sh
datasets/naps/preprocessing/run_all.sh
datasets/isruc_s1/preprocessing/run_all.sh

# 2. modelling: 100%-label CV first (its folds and settings are reused), then the 25%-label experiment
datasets/isleeps/modelling/run_all.sh
datasets/naps/modelling/run_all.sh
datasets/isruc_s1/modelling/run_all.sh
datasets/sleepedf20/modelling/run_all.sh

# 3. analyses, tables and figures
analysis/run_all.sh
```

Training runs fold by fold, writes `config.json` (settings, code and data hashes) before training, refuses to mix
settings in one output folder, and resumes interrupted folds exactly. The published runs used one GPU
(NVIDIA RTX A5000). `isruc_cv.py`, `sleepedf_cv.py` and `sleepedf_teacher4.py` accept `--check` to load the data and print
folds and scaling without training.

Outputs per cohort (`results/<cohort>/`): per-fold `test_metrics.json` and `test_predictions.npz`, checkpoints,
`SUMMARY.md`/`summary.json`; the 25%-label runs also write `stats.json` (paired per-patient Wilcoxon tests).
`paper/make_tables.py` reads only these files, so every number in the paper tables is regenerated from saved
predictions.

## Notes

- Student input: signals in µV, resampled once to 100 Hz (`scipy.signal.resample_poly`, no other filtering),
  z-scored per channel with training-patient statistics stored in each checkpoint.
- Reported gains are means of paired per-patient macro-F1 differences (stages present in each patient), not
  differences of fold means.
- In Sleep-EDF-20 the chin EMG is stored as a 1 Hz envelope; `sleepedf_teacher4.py` interpolates it to 100 Hz,
  which differs from the raw EMG SleepGPT was pretrained on.

## Licence and citation

Licence: to be added by the authors. SleepGPT and the datasets are subject to their own terms.

```bibtex
@inproceedings{maiti2027gcvtad,
  title     = {Towards wearable-scale, transition-aware sleep staging for clinical populations},
  author    = {Maiti, Suvadeep and Sinha, Keya and Raju S, Bapi},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {submitted}
}
```
