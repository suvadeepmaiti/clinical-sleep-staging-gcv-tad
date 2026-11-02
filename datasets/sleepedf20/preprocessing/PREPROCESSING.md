# Training-ready Sleep-EDF-20

The generated data are in `data/preprocessed/sleepedf20/`: **one `.npz` file per
recording/night**, covering the 20 subjects SC400–SC419 and all 39 nights.
Original EDFs remain in `data/raw/sleep-cassette/` (download_sleepedf20.py).

The completed run contains **42,307 epochs**: W 8,284; N1 2,804; N2 17,799;
N3 5,703; REM 7,717. It excludes 61 movement epochs within the retained windows.
The example splits contain 29,558 training, 7,244 validation, and 5,505 test epochs.

## Processing protocol

1. Match PSG and hypnogram by the full six-character recording ID. Annotation
   suffixes vary (EC, EH, EJ, EP, EU, EV); all are handled.
2. Extract **EEG Fpz-Cz** and **EOG horizontal**, in that order. Both are sampled
   at their native **100 Hz** and converted from EDF digital values to **µV**.
3. Align labels using annotation onset times and recording start times. Divide
   the original PSG time grid into non-overlapping **30-second epochs** with
   **3,000 samples per channel**.
4. Find the first and last epoch scored N1, N2, N3, or REM. Keep from **60 epochs
   before the first** through **60 epochs after the last**, clipped to available
   recording time. This is 30 minutes on either side of the scored sleep period.
   Keep wake periods between those boundaries. Do not trim around every awakening.
5. Merge original R&K stages 3 and 4 into N3. This gives the common five-class
   mapping; the original annotations are not independently rescored under AASM.

   | Integer | Class | Original annotation |
   |---|---|---|
   | 0 | W | Sleep stage W |
   | 1 | N1 | Sleep stage 1 |
   | 2 | N2 | Sleep stage 2 |
   | 3 | N3 | Sleep stage 3 or 4 |
   | 4 | REM | Sleep stage R |

6. Exclude movement, unknown, and unannotated epochs. These are **not relabeled
   as wake**. Crop using original elapsed time before dropping invalid epochs;
   deleting an epoch does not shift the crop boundary.
7. Save calibrated signals as `float32` and class labels as `int64`. There is
   **no extra filtering, resampling, artifact rejection, or class balancing**.
   Saved `x` is not normalized, preserving amplitudes for different models.

This follows the common Sleep-EDF epoching, five-stage mapping, and 30-minute
wake-margin approach in the authors' [DeepSleepNet preprocessing](https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py)
and [TinySleepNet preprocessing](https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py),
extended to two channels with explicit time-gap metadata. There is no single
universal filtering or normalization recipe. This implementation anchors the
crop to **valid sleep stages only**, whereas some reference implementations
count unknown/movement as non-wake or remove epochs before cropping. Those
differences can change epoch counts, so use `summary.json` for this run's totals.

For this dataset, removing invalid epochs before cropping produces 42,308 epochs.
The extra wake epoch is original epoch 1953 in SC4092 (starting at 58,590 seconds).
Deleting a movement epoch first shifts the selected endpoint beyond the requested
30-minute margin. The strict elapsed-time crop implemented here excludes that
extra epoch and retains **42,307**. All non-wake class counts remain identical.

## Load for training

Install the small dependencies in your chosen Python environment:

```bash
python -m pip install -r requirements.txt
```

From this project folder:

```python
from sleepedf_data import load_split

train = load_split("train")
val = load_split("val")
test = load_split("test")

X_train, y_train = train["x"], train["y"]
# X_train: float32 (N, 2, 3000), channels = [EEG Fpz-Cz, EOG horizontal]
# y_train: int64   (N,), values 0–4
```

`load_split` applies per-channel z-score normalization using **only the training
subjects' mean and standard deviation**, saved in `splits.json`. Validation and
test use those same statistics. Pass `normalize=False` to obtain microvolts.
Loading an entire split concatenates its files in memory; individual nightly
files can be loaded separately when RAM is limited.

The example split uses **14 train / 3 validation / 3 test subjects**, seed 42.
It is a convenience split, **not a published benchmark split**. Both nights of
each person always belong to the same split. For benchmark evaluation, use
subject-wise cross-validation and refit normalization on each training fold.
Do not randomly divide epochs from the same person across train and test.

Optional PyTorch conversion (install PyTorch separately for your hardware):

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=64, shuffle=True,
)
# A Conv1d model should accept in_channels=2.
```

## Load one recording or make custom folds

```python
import numpy as np

with np.load("data/preprocessed/sleepedf20/SC4001.npz", allow_pickle=False) as data:
    x, y = data["x"], data["y"]  # x is in µV
    epoch_index = data["epoch_index"]
    segment_id = data["segment_id"]

from sleepedf_data import fit_standardizer, load_subjects

folder = "preprocessed_sleepedf20"
training_subjects = ["SC400", "SC401"]  # choose your actual training fold
stats = fit_standardizer(folder, training_subjects)
training = load_subjects(training_subjects, folder, normalization=stats)
held_out = load_subjects(["SC402"], folder, normalization=stats)
```

For sequence models and transition estimation, only join epochs from the **same
recording and segment_id**, with consecutive `epoch_index` values. Excluded
epochs leave time gaps; concatenated array positions alone do not imply temporal
adjacency. Segment IDs are local to each recording.

## Files and metadata

- `SC4001.npz`, etc.: `x`, `y`, `subject` (e.g. `SC400`), `recording` (e.g.
  `SC4001`), `night`, `channels`, `units`, `fs`, `epoch_seconds`, `epoch_index`,
  `onset_seconds` (relative to original PSG start), `segment_id`, start time,
  crop bounds, sleep bounds, and original filenames. No pickle is needed.
- `records.csv`: per-night stage counts, crop bounds, exclusion counts, and
  SHA-256 hashes for source EDFs and generated NPZs.
- `summary.json`: overall counts, processing choices, and dependency versions.
- `splits.json`: example subject split and training-only normalization statistics.
- `validation.json`: completed integrity, annotation alignment, crop-boundary,
  calibrated signal sample, and split checks for all 39 recordings.

## Reproduce

```bash
python preprocess_sleepedf20.py --out-dir preprocessed_sleepedf20_new
python -m unittest -v test_preprocessing
```

The script requires an empty output directory and does not overwrite existing
processed datasets. Default paths are resolved relative to the script directory.
Loading existing files only requires NumPy; regenerating them also needs pyEDFlib.
The initial run used an isolated Linux environment at `/tmp/sleepedf20-venv`;
the data files are portable to Windows, Linux, and other NumPy environments.
