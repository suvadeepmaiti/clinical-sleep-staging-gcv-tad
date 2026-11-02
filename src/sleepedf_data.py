"""NumPy loaders for the preprocessed two-channel Sleep-EDF-20 dataset."""
import json
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "preprocessed_sleepedf20"


def _paths(data_dir, subjects):
    subjects = list(subjects)
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("Provide a nonempty list of unique subject IDs, e.g. ['SC400']")
    paths = sorted(p for p in Path(data_dir).glob("SC4*.npz") if p.stem[:5] in subjects)
    missing = set(subjects) - {p.stem[:5] for p in paths}
    if missing:
        raise ValueError(f"Subjects not found: {sorted(missing)}")
    return paths


def fit_standardizer(data_dir, train_subjects):
    """Fit two channel-wise means/stds using training subjects exclusively."""
    sums = np.zeros(2, dtype=np.float64)
    squares = np.zeros(2, dtype=np.float64)
    count = 0
    for path in _paths(data_dir, train_subjects):
        with np.load(path, allow_pickle=False) as data:
            x = data["x"]
            for c in range(2):
                values = x[:, c, :].astype(np.float64)
                sums[c] += values.sum()
                squares[c] += np.square(values).sum()
            count += len(x) * x.shape[-1]
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    if np.any(std < 1e-8):
        raise ValueError("Cannot standardize a constant channel")
    return {"method": "Per-channel z-score fitted on training subjects only",
            "fit_subjects": sorted(train_subjects), "mean_uV": mean.tolist(),
            "std_uV": std.tolist(), "samples_per_channel": count}


def load_subjects(subjects, data_dir=DEFAULT_DATA_DIR, normalization=None):
    """Return arrays in recording/time order; x is (N, 2, 3000), y is (N,).

    With normalization=None, x contains original microvolts. Otherwise pass
    the result of fit_standardizer(data_dir, TRAIN_SUBJECTS). Never fit on
    validation/test subjects. Sequence windows must share recording and
    segment_id; epoch_index preserves positions before exclusion.
    """
    parts = {key: [] for key in ("x", "y", "subject", "recording", "epoch_index", "segment_id")}
    for path in _paths(data_dir, subjects):
        with np.load(path, allow_pickle=False) as data:
            x, y = data["x"], data["y"]
            if normalization is not None:
                mean = np.asarray(normalization["mean_uV"], dtype=np.float32).reshape(1, 2, 1)
                std = np.asarray(normalization["std_uV"], dtype=np.float32).reshape(1, 2, 1)
                if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                    raise ValueError("Invalid normalization statistics")
                x -= mean
                x /= std
            parts["x"].append(x)
            parts["y"].append(y)
            for key in ("subject", "recording"):
                parts[key].append(np.full(len(y), data[key].item()))
            for key in ("epoch_index", "segment_id"):
                parts[key].append(data[key])
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def load_split(split="train", data_dir=DEFAULT_DATA_DIR, normalize=True):
    """Load the supplied example subject-disjoint split, standardized by default."""
    metadata = json.loads((Path(data_dir) / "splits.json").read_text())
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val', or 'test'")
    subjects = metadata["subjects"]
    if (set(subjects["train"]) & set(subjects["val"]) or
            set(subjects["train"]) & set(subjects["test"]) or
            set(subjects["val"]) & set(subjects["test"])):
        raise ValueError("Subjects overlap between splits")
    normalization = metadata["normalization"] if normalize else None
    if normalize and set(normalization["fit_subjects"]) != set(subjects["train"]):
        raise ValueError("Normalization was not fitted on these training subjects")
    return load_subjects(subjects[split], data_dir, normalization)
