"""Prepare Sleep-EDF-20 Fpz-Cz EEG + horizontal EOG for five-stage training."""
import sys
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyedflib

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
CHANNELS = ("EEG Fpz-Cz", "EOG horizontal")
STAGES = ("W", "N1", "N2", "N3", "REM")
FS = 100
EPOCH_SECONDS = 30
SAMPLES = FS * EPOCH_SECONDS
LABEL_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
    "Movement time": -2, "Sleep stage ?": -1,
}


def epoch_labels(onsets, durations, descriptions, n_epochs, offset=0):
    """Map annotations to the original PSG epoch grid; reject ambiguous timing."""
    labels = np.full(n_epochs, -3, dtype=np.int8)  # -3 means unannotated
    for onset, duration, description in zip(onsets, durations, descriptions):
        if description not in LABEL_MAP:
            raise ValueError(f"Unexpected annotation: {description!r}")
        start = (float(onset) + offset) / EPOCH_SECONDS
        stop = (float(onset) + offset + float(duration)) / EPOCH_SECONDS
        if duration <= 0 or not np.allclose([start, stop], np.rint([start, stop]), atol=1e-7, rtol=0):
            raise ValueError(f"Annotation not aligned to 30 seconds: {onset}, {duration}, {offset}")
        lo, hi = max(0, round(start)), min(n_epochs, round(stop))
        if hi <= lo:  # Annotations may extend beyond the PSG recording.
            continue
        if np.any(labels[lo:hi] != -3):
            raise ValueError("Overlapping annotations")
        labels[lo:hi] = LABEL_MAP[description]
    return labels


def select_epochs(labels, wake_minutes=30):
    """Crop in original elapsed time, then exclude movement/unknown/unscored."""
    sleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    if not len(sleep):
        raise ValueError("No scored sleep in this recording")
    margin = wake_minutes * 60 / EPOCH_SECONDS
    if margin < 0 or not float(margin).is_integer():
        raise ValueError("Wake margin must be a nonnegative multiple of 30 seconds")
    start = max(0, int(sleep[0]) - int(margin))
    stop = min(len(labels), int(sleep[-1]) + 1 + int(margin))
    indices = np.arange(start, stop, dtype=np.int64)
    indices = indices[(labels[indices] >= 0) & (labels[indices] <= 4)]
    return indices, start, stop, int(sleep[0]), int(sleep[-1])


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_pair(psg_path, hyp_path, out_dir):
    recording = psg_path.name[:6]
    subject = recording[:5]
    with pyedflib.EdfReader(str(psg_path)) as psg, pyedflib.EdfReader(str(hyp_path)) as hyp:
        signal_labels = [label.strip() for label in psg.getSignalLabels()]
        channel_indices = [signal_labels.index(name) for name in CHANNELS]
        sizes = [int(psg.getNSamples()[i]) for i in channel_indices]
        if len(set(sizes)) != 1 or sizes[0] % SAMPLES:
            raise ValueError(f"Unequal channel lengths or incomplete epochs: {sizes}")
        for i in channel_indices:
            if psg.getSampleFrequency(i) != FS:
                raise ValueError(f"Unexpected sample rate: {psg_path}, {signal_labels[i]}")
            if psg.getPhysicalDimension(i).strip() not in ("uV", "µV", "μV"):
                raise ValueError(f"Expected microvolts: {psg_path}, {signal_labels[i]}")
        n_epochs = sizes[0] // SAMPLES
        start_datetime = psg.getStartdatetime()
        offset = (hyp.getStartdatetime() - start_datetime).total_seconds()
        labels = epoch_labels(*hyp.readAnnotations(), n_epochs, offset)
        indices, start, stop, first_sleep, last_sleep = select_epochs(labels)
        # Read only the cropped interval, in physical units using EDF calibration.
        x = np.empty((len(indices), len(CHANNELS), SAMPLES), dtype=np.float32)
        for c, i in enumerate(channel_indices):
            signal = psg.readSignal(i, start=start * SAMPLES, n=(stop - start) * SAMPLES)
            x[:, c, :] = signal.reshape(-1, SAMPLES)[indices - start]
    y = labels[indices].astype(np.int64)
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite samples in {recording}")
    segment_id = np.cumsum(np.r_[0, np.diff(indices) != 1]).astype(np.int32)
    path = out_dir / f"{recording}.npz"
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, x=x, y=y, subject=subject, recording=recording,
            night=int(recording[-1]), channels=np.array(CHANNELS), units="uV",
            fs=FS, epoch_seconds=EPOCH_SECONDS, epoch_index=indices,
            onset_seconds=indices * EPOCH_SECONDS, segment_id=segment_id,
            start_datetime=start_datetime.isoformat(),
            crop_start_epoch=start, crop_stop_epoch_exclusive=stop,
            first_sleep_epoch=first_sleep, last_sleep_epoch=last_sleep,
            source_psg=psg_path.name, source_hypnogram=hyp_path.name,
        )
    temporary.replace(path)
    return {
        "file": path.name, "subject": subject, "recording": recording,
        "night": int(recording[-1]), "original_epochs": n_epochs,
        "crop_start_epoch": start, "crop_stop_epoch_exclusive": stop,
        "first_sleep_epoch": first_sleep, "last_sleep_epoch": last_sleep,
        "trimmed_outside_window": start + n_epochs - stop,
        "excluded_movement": int(np.sum(labels[start:stop] == -2)),
        "excluded_unknown": int(np.sum(labels[start:stop] == -1)),
        "excluded_unannotated": int(np.sum(labels[start:stop] == -3)),
        "retained_epochs": len(y), "continuous_segments": int(segment_id[-1]) + 1,
        **{stage: int(np.sum(y == i)) for i, stage in enumerate(STAGES)},
        "psg_sha256": file_hash(psg_path), "hypnogram_sha256": file_hash(hyp_path),
        "output_sha256": file_hash(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=repo_paths.RAW["sleepedf20"])
    parser.add_argument("--out-dir", type=Path, default=repo_paths.PREPROCESSED["sleepedf20"])
    args = parser.parse_args()
    files = sorted(p for p in args.raw_dir.glob("*-PSG.edf")
                   if re.fullmatch(r"SC4[01][0-9][12]E0-PSG\.edf", p.name))
    expected = {f"SC4{s:02d}{n}" for s in range(20) for n in (1, 2)} - {"SC4132"}
    if {p.name[:6] for p in files} != expected or len(files) != 39:
        raise ValueError("Expected the complete Sleep-EDF-20 subset (39 nights, SC400–SC419)")
    pairs = []
    for psg in files:
        matches = list(args.raw_dir.glob(f"{psg.name[:6]}*-Hypnogram.edf"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one hypnogram for {psg.name}: {matches}")
        pairs.append((psg, matches[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output folder must be empty: {args.out_dir}. Choose a new --out-dir.")
    records = []
    for i, (psg, hyp) in enumerate(pairs, 1):
        row = process_pair(psg, hyp, args.out_dir)
        records.append(row)
        print(f"[{i:02d}/39] {row['recording']}: {row['retained_epochs']} epochs", flush=True)
    with (args.out_dir / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "dataset": "Sleep-EDF-20", "source": "https://physionet.org/content/sleep-edfx/1.0.0/",
        "channels": CHANNELS, "shape": "(epochs, channels, samples)",
        "fs": FS, "epoch_seconds": EPOCH_SECONDS, "units": "uV",
        "x_dtype": "float32", "y_dtype": "int64", "label_names": STAGES,
        "wake_margin_minutes": 30, "crop_basis": "First and last valid N1/N2/N3/REM epoch on original time grid",
        "excluded_labels": ["Movement time", "Sleep stage ?", "unannotated"],
        "filtering": "No additional digital filtering", "resampling": "None (native 100 Hz)",
        "saved_normalization": "None (calibrated microvolts); loader supports training-only standardization",
        "subjects": 20, "recordings": len(records),
        "retained_epochs": sum(r["retained_epochs"] for r in records),
        "class_counts": {s: sum(r[s] for r in records) for s in STAGES},
        "excluded_within_window": {reason: sum(r[reason] for r in records) for reason in
                                   ["excluded_movement", "excluded_unknown", "excluded_unannotated"]},
        "versions": {"numpy": np.__version__, "pyedflib": pyedflib.__version__},
        "preprocessing_script_sha256": file_hash(Path(__file__)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from sleepedf_data import fit_standardizer
    subjects = sorted({r["subject"] for r in records})
    shuffled = np.random.default_rng(42).permutation(subjects).tolist()
    splits = {"train": sorted(shuffled[:14]), "val": sorted(shuffled[14:17]), "test": sorted(shuffled[17:])}
    split_document = {
        "description": "Example 14/3/3 subject-disjoint holdout; not a canonical benchmark split",
        "seed": 42, "subjects": splits,
        "normalization": fit_standardizer(args.out_dir, splits["train"]),
    }
    (args.out_dir / "splits.json").write_text(json.dumps(split_document, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
