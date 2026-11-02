"""iSLEEPS data loading, montage-stratified patient folds and gap-aware context windows."""
import csv
import sys
from pathlib import Path

import numpy as np

from protocol import STAGES, VARIANTS, metrics, normalization, sha256, temporal_indices, write_json  # noqa: E402,F401

CHANNEL_SETS = {"c4": ["C4-M1"]}  # C4:M1, or C4:A1 in auricular-labelled files
N_SUBJECTS = 99


def load_data(folder, channels=("C4-M1",)):
    """Read selected channels from the preprocessed iSLEEPS NPZs (one per patient)."""
    folder, channels = Path(folder), list(channels)
    with (folder / "records.csv").open() as stream:
        montage = {r["subject"]: r["montage"] for r in csv.DictReader(stream)}
    paths = sorted(folder.glob("SN*.npz"), key=lambda p: int(p.stem[2:]))
    if len(paths) != N_SUBJECTS or {p.stem for p in paths} != set(montage):
        raise ValueError(f"Expected {N_SUBJECTS} preprocessed iSLEEPS NPZ files matching records.csv")
    xs, ys, subjects, epochs, segments, records, sources = [], [], [], [], [], [], {}
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            names = d["channels"].tolist()
            selected = [names.index(c) for c in channels]
            if not d["channel_present"][selected].all():
                raise ValueError(f"{path.stem} lacks one of {channels}")
            x = np.ascontiguousarray(d["x"][:, selected, :], dtype=np.float32)
            if len(channels) == 1:
                x = x[:, 0, :].copy()
            y = d["y"].astype(np.int64)
            n = len(y)
            if d["fs"].item() != 100 or not np.isfinite(x).all() or x.shape[0] != n:
                raise ValueError(f"Invalid signal data: {path}")
            subject = d["subject"].item()
            assert subject == path.stem
            sources[subject] = d["source_channels"][selected].tolist()
            xs.append(x); ys.append(y); subjects.append(np.full(n, subject))
            epochs.append(d["epoch_index"]); segments.append(d["segment_id"])
            records.append({"recording": subject, "subject": subject, "start": offset, "stop": offset + n,
                            "sha256": sha256(path), "montage": montage[subject],
                            "sum": x.sum(axis=(0, -1), dtype=np.float64).tolist(),
                            "sum_squares": np.square(x.astype(np.float64)).sum(axis=(0, -1)).tolist(),
                            "sample_count": int(n * 3000)})
            offset += n
    data = {k: np.concatenate(v) for k, v in zip(
        ["x", "y", "subject", "epoch_index", "segment_id"], [xs, ys, subjects, epochs, segments])}
    data["recording"] = data["subject"]  # one night per patient
    data["records"], data["channels"], data["source_channels"] = records, channels, sources
    return data


def _shuffled_strata(subjects, strata, rng):
    return {stratum: rng.permutation(sorted(s for s in subjects if strata[s] == stratum)).tolist()
            for stratum in sorted({strata[s] for s in subjects})}


def make_folds(subjects, strata, n_folds=10, val_fraction=0.1, seed=42):
    """Patient-disjoint K folds stratified by montage; inner validation drawn from the rest."""
    subjects = sorted(map(str, subjects))
    if len(set(subjects)) != len(subjects) or set(subjects) != set(strata):
        raise ValueError("Subjects must be distinct and all have a stratum")
    # Deal each shuffled stratum round-robin with one running counter: every fold receives
    # floor/ceil of each stratum and fold sizes differ by at most one.
    order = [s for members in _shuffled_strata(subjects, strata, np.random.default_rng(seed)).values()
             for s in members]
    assignment = {s: i % n_folds for i, s in enumerate(order)}
    folds = []
    for k in range(n_folds):
        test = sorted(s for s in subjects if assignment[s] == k)
        remaining = [s for s in subjects if assignment[s] != k]
        n_val = int(round(val_fraction * len(remaining)))
        groups = _shuffled_strata(remaining, strata, np.random.default_rng(seed + k + 1))
        quota = {g: n_val * len(m) / len(remaining) for g, m in groups.items()}
        counts = {g: int(q) for g, q in quota.items()}
        for g in sorted(quota, key=lambda g: (counts[g] - quota[g], g))[:n_val - sum(counts.values())]:
            counts[g] += 1  # largest remainder
        val = sorted(s for g, members in groups.items() for s in members[:counts[g]])
        train = sorted(set(remaining) - set(val))
        assert not set(train) & set(val) and not set(test) & set(train + val)
        folds.append({"fold": k + 1, "train": train, "val": val, "test": test})
    assert sorted(s for f in folds for s in f["test"]) == subjects
    return folds


def context_windows(data, window=9, target=4):
    """(N, window) global indices centred on each epoch; edges repeat the nearest epoch.

    Windows never leave a continuous run (same recording, consecutive epoch_index), so they
    never cross removed artefact epochs, recordings or subjects.
    """
    n = len(data["y"])
    breaks = np.flatnonzero((np.diff(data["epoch_index"]) != 1) |
                            (data["recording"][1:] != data["recording"][:-1])) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, n]
    run_start = np.repeat(starts, stops - starts)
    run_stop = np.repeat(stops, stops - starts)
    windows = np.arange(n)[:, None] + np.arange(window)[None, :] - target
    windows = np.clip(windows, run_start[:, None], run_stop[:, None] - 1)
    assert np.all(windows[:, target] == np.arange(n))
    return windows.astype(np.int64)
