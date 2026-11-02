"""Subject-disjoint folds, temporal indexing and fixed-five-class metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
CHANNEL_SETS = {"eeg": ["EEG Fpz-Cz"], "eeg_eog": ["EEG Fpz-Cz", "EOG horizontal"]}
VARIANTS = {
    "ce_only": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "no_kd": {"alpha": 0.3, "beta": 0.0, "gamma": 0.2},
    "full_tad": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
}


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_folds(subjects, seed=42, val_count=2):
    subjects = sorted(map(str, subjects))
    if len(subjects) != 20 or len(set(subjects)) != 20:
        raise ValueError("LOSO requires exactly 20 distinct subjects")
    folds = []
    for fold, test in enumerate(subjects, 1):
        remaining = [s for s in subjects if s != test]
        val = sorted(np.random.default_rng(seed + fold).choice(remaining, val_count, replace=False).tolist())
        train = sorted(set(remaining) - set(val))
        assert not set(train) & set(val) and test not in train + val
        folds.append({"fold": fold, "train": train, "val": val, "test": [test]})
    return folds


def load_data(folder, channels=("EEG Fpz-Cz",)):
    """Read selected signals; preserve epoch identities and recording boundaries."""
    channels = list(channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Channels must be nonempty and unique")
    xs, ys, subjects, recordings, epochs, segments, records = [], [], [], [], [], [], []
    offset = 0
    paths = sorted(Path(folder).glob("SC4*.npz"))
    if len(paths) != 39:
        raise ValueError("Expected 39 preprocessed nightly NPZ files")
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            selected = [d["channels"].tolist().index(c) for c in channels]
            x = np.ascontiguousarray(d["x"][:, selected, :], dtype=np.float32)
            if len(channels) == 1:
                x = x[:, 0, :].copy()
            y = d["y"].astype(np.int64)
            n = len(y)
            shape = (n, 3000) if len(channels) == 1 else (n, len(channels), 3000)
            if x.shape != shape or d["fs"].item() != 100 or not np.isfinite(x).all():
                raise ValueError(f"Invalid signal data: {path}")
            subject, recording = d["subject"].item(), d["recording"].item()
            assert subject == recording[:5] and recording == path.stem
            xs.append(x); ys.append(y)
            subjects.append(np.full(n, subject)); recordings.append(np.full(n, recording))
            epochs.append(d["epoch_index"]); segments.append(d["segment_id"])
            records.append({"recording": recording, "subject": subject, "start": offset,
                            "stop": offset + n, "sha256": sha256(path),
                            "sum": x.sum(axis=(0, -1), dtype=np.float64).tolist(),
                            "sum_squares": np.square(x.astype(np.float64)).sum(axis=(0, -1)).tolist(),
                            "sample_count": int(n * 3000)})
            offset += n
    data = {k: np.concatenate(v) for k, v in zip(
        ["x", "y", "subject", "recording", "epoch_index", "segment_id"],
        [xs, ys, subjects, recordings, epochs, segments])}
    assert len(np.unique(data["subject"])) == 20
    data["records"] = records
    data["channels"] = channels
    return data


def normalization(data, train_subjects):
    records = [r for r in data["records"] if r["subject"] in train_subjects]
    n = sum(r["sample_count"] for r in records)
    mean = sum(np.asarray(r["sum"]) for r in records) / n
    std = (sum(np.asarray(r["sum_squares"]) for r in records) / n - mean * mean) ** .5
    if not np.all(np.isfinite(std) & (std > 0)):
        raise ValueError("Invalid training-only signal standard deviation")
    return {"mean_uV": mean.tolist(), "std_uV": std.tolist(),
            "fit_subjects": sorted(train_subjects)}


def temporal_indices(data, subjects, sequence_length=20):
    """All epoch indices, valid next pairs, and variable-length contiguous sequences."""
    selected = np.flatnonzero(np.isin(data["subject"], subjects))
    groups = {}
    pair_starts = []
    for recording in np.unique(data["recording"][selected]):
        indices = selected[data["recording"][selected] == recording]
        breaks = np.flatnonzero(np.diff(data["epoch_index"][indices]) != 1) + 1
        for continuous in np.split(indices, breaks):
            if len(continuous) > 1:
                pair_starts.extend(continuous[:-1])
            for start in range(0, len(continuous), sequence_length):
                seq = continuous[start:start + sequence_length]
                groups.setdefault(len(seq), []).append(seq)
    groups = {length: np.stack(values) for length, values in groups.items()}
    pairs = np.asarray(pair_starts, dtype=np.int64)
    assert np.array_equal(np.sort(np.concatenate([v.ravel() for v in groups.values()])), selected)
    if len(pairs):
        assert np.all(data["recording"][pairs] == data["recording"][pairs + 1])
        assert np.all(data["epoch_index"][pairs + 1] == data["epoch_index"][pairs] + 1)
    return selected, pairs, groups


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    if y.shape != pred.shape or not len(y) or np.any((y < 0) | (y > 4) | (pred < 0) | (pred > 4)):
        raise ValueError("Metrics require matching nonempty vectors with labels 0–4")
    cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(5, dtype=np.float64), where=denom != 0)
    accuracy = float(np.trace(cm) / cm.sum())
    expected = float(np.dot(cm.sum(0), cm.sum(1)) / float(cm.sum()) ** 2)
    return {"accuracy": accuracy, "macro_f1": float(f1.mean()),
            "kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else 0.0,
            **{f"f1_{name}": float(v) for name, v in zip(STAGES, f1)},
            "n_epochs": len(y), "confusion_matrix": cm.tolist()}
