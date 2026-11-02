"""Preprocess ISRUC-Sleep Cohort I (100 sleep-disorder patients, 2 scorers) into the iSLEEPS/NAPS NPZ format.

Per recording N (<raw ISRUC-S1>/N/N.rec, hypnograms N_1.txt / N_2.txt; no other file is read):
- Channels -> C3-M2, C4-M1, O1-M2, O2-M1, F3-M2, F4-M1, E1-M2, E2-M1, EMG (chin). Sources: '-M1/-M2'
  derivations, or '-A1/-A2' (A = mastoid = M), or monopolar pairs (recording 40: C3 - A2, LOC - A2, ...);
  chin EMG is 'X1' or, in the numbered layout, '24' (identified by spectrum: ECG is X2/25, legs X3-X4/26-27).
  Missing -> NaN. Every channel resampled once from its native rate (200 Hz) to 100 Hz (resample_poly).
  No filtering, no re-scaling, no normalization.
- Labels: scorer 1 is the reference (0 W, 1 N1, 2 N2, 3 N3, 5 REM -> 0..4); scorer 2 kept as y_scorer2.
  Epoch k covers [30k, 30k + 30) s from the EDF start; epochs beyond the EDF or the scorer-1 file are dropped.
- Cropping (Sleep-EDF convention): keep from 30 min (60 epochs) before the first to 30 min after the last
  scorer-1 sleep epoch (N1/N2/N3/REM), clipped to the recording.
"""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
from naps_edf import read_edf, signal  # noqa: E402

FS, EPOCH_SECONDS, WAKE_MARGIN_EPOCHS = 100, 30, 60
CHANNELS = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "F3-M2", "F4-M1", "E1-M2", "E2-M1", "EMG"]
SOURCES = {"C3-M2": ["C3-M2", "C3-A2", ("C3", "A2")], "C4-M1": ["C4-M1", "C4-A1", ("C4", "A1")],
           "O1-M2": ["O1-M2", "O1-A2", ("O1", "A2")], "O2-M1": ["O2-M1", "O2-A1", ("O2", "A1")],
           "F3-M2": ["F3-M2", "F3-A2", ("F3", "A2")], "F4-M1": ["F4-M1", "F4-A1", ("F4", "A1")],
           "E1-M2": ["E1-M2", "LOC-A2", ("LOC", "A2")], "E2-M1": ["E2-M1", "ROC-A1", ("ROC", "A1")],
           "EMG": ["X1", "24"]}
LABELS = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
STAGES = ("W", "N1", "N2", "N3", "REM")


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_labels(path):
    return np.array([LABELS.get(int(v), -1) for v in path.read_text().split()], dtype=np.int64)


def channel(edf, options):
    """(native-rate physical signal, fs, source name, clip mask) for the first available option, else None."""
    labels = edf["labels"]
    for opt in options:
        if isinstance(opt, tuple):
            if all(o in labels for o in opt):
                (a, fa), (b, fb) = (signal(edf, labels.index(o)) for o in opt)
                assert fa == fb
                clip = clipped(edf, labels.index(opt[0])) | clipped(edf, labels.index(opt[1]))
                return a - b, fa, f"{opt[0]}-{opt[1]} (computed)", clip
        elif opt in labels:
            i = labels.index(opt)
            x, fs = signal(edf, i)
            return x, fs, opt, clipped(edf, i)
    return None


def clipped(edf, i):
    lo, hi = edf["offsets"][i], edf["offsets"][i + 1]
    digital = edf["records"][:, lo:hi].reshape(-1)
    return (digital <= edf["dmin"][i]) | (digital >= edf["dmax"][i])


def process(n, root, out_dir):
    folder = root / str(n)
    subject = f"ISRUC1_{n:03d}"
    edf = read_edf(folder / f"{n}.rec")
    y1, y2 = read_labels(folder / f"{n}_1.txt"), read_labels(folder / f"{n}_2.txt")
    seconds = edf["n_records"] * edf["record_seconds"]
    n_edf = int(seconds // EPOCH_SECONDS)
    n_ep = min(n_edf, len(y1))
    y1 = y1[:n_ep]
    y2 = np.r_[y2[:n_ep], np.full(max(0, n_ep - len(y2)), -1)]
    sleep = np.flatnonzero(np.isin(y1, (1, 2, 3, 4)))
    start, stop = max(0, sleep[0] - WAKE_MARGIN_EPOCHS), min(n_ep - 1, sleep[-1] + WAKE_MARGIN_EPOCHS)
    indices = np.flatnonzero((np.arange(n_ep) >= start) & (np.arange(n_ep) <= stop) & (y1 >= 0))
    x = np.full((len(indices), len(CHANNELS), FS * EPOCH_SECONDS), np.nan, dtype=np.float32)
    clip = np.zeros((len(indices), len(CHANNELS)), dtype=np.float32)
    present = np.zeros(len(CHANNELS), dtype=bool)
    sources, native = [""] * len(CHANNELS), np.zeros(len(CHANNELS), dtype=np.int64)
    for c, name in enumerate(CHANNELS):
        got = channel(edf, SOURCES[name])
        if got is None:
            continue
        sig, fs, src, cmask = got
        spe = int(round(fs * EPOCH_SECONDS))
        ratio = Fraction(FS, int(round(fs))).limit_denominator(1000)
        down = resample_poly(sig[:n_ep * spe], ratio.numerator, ratio.denominator).astype(np.float32)
        x[:, c] = down[:n_ep * FS * EPOCH_SECONDS].reshape(n_ep, -1)[indices]
        clip[:, c] = cmask[:n_ep * spe].reshape(n_ep, spe).mean(1)[indices]
        present[c], sources[c], native[c] = True, src, int(round(fs))
    labels_kept, labels2_kept = y1[indices], y2[indices]
    segment_id = np.cumsum(np.r_[0, np.diff(indices) != 1])
    montage = ("monopolar" if any("computed" in s for s in sources) else
               "A1/A2" if any(s.endswith(("-A1", "-A2")) for s in sources) else "M1/M2")
    path = out_dir / f"{subject}.npz"
    temporary = path.with_suffix(".tmp.npz")
    with open(temporary, "wb") as stream:
        np.savez(stream, x=x, y=labels_kept, y_scorer2=labels2_kept, subject=subject, recording=subject,
                 channels=np.array(CHANNELS), channel_present=present, source_channels=np.array(sources),
                 native_fs=native, units="uV", fs=FS, epoch_seconds=EPOCH_SECONDS, epoch_index=indices,
                 onset_seconds=indices * EPOCH_SECONDS, segment_id=segment_id, clip_fraction=clip, montage=montage,
                 edf_start_time=edf["start_time"], source_psg=f"{n}/{n}.rec",
                 source_hypnograms=np.array([f"{n}/{n}_1.txt", f"{n}/{n}_2.txt"]))
    temporary.replace(path)
    both = labels2_kept >= 0
    agree = float(np.mean(labels_kept[both] == labels2_kept[both])) if both.any() else float("nan")
    counts = np.bincount(labels_kept, minlength=5)
    return {"file": path.name, "subject": subject, "isruc_id": n, "montage": montage,
            "recording_seconds": seconds, "edf_epochs": n_edf, "scorer1_epochs": int(len(read_labels(folder / f"{n}_1.txt"))),
            "scorer2_epochs": int(len(read_labels(folder / f"{n}_2.txt"))), "first_sleep_epoch": int(sleep[0]),
            "last_sleep_epoch": int(sleep[-1]), "crop_start_epoch": int(start), "crop_stop_epoch": int(stop),
            "excluded_outside_crop": int(n_ep - (stop - start + 1)), "excluded_invalid_label": int(np.sum(y1[start:stop + 1] < 0)),
            "retained_epochs": int(len(indices)), "continuous_segments": int(segment_id[-1] + 1),
            **{s: int(k) for s, k in zip(STAGES, counts)},
            "scorer_agreement": round(agree, 4),
            "channels_present": ";".join(c for c, p in zip(CHANNELS, present) if p),
            "missing_channels": ";".join(c for c, p in zip(CHANNELS, present) if not p),
            "eog_source": sources[CHANNELS.index("E1-M2")], "emg_source": sources[CHANNELS.index("EMG")],
            "psg_sha256": edf["sha256"], "output_sha256": file_hash(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_paths.RAW["isruc"])
    parser.add_argument("--out-dir", type=Path, default=repo_paths.PREPROCESSED["isruc"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.name) for p in args.root.iterdir() if p.is_dir() and p.name.isdigit())
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(process, ids, [args.root] * len(ids), [args.out_dir] * len(ids)))
    with (args.out_dir / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    total = {s: sum(r[s] for r in records) for s in STAGES}
    summary = {"dataset": "ISRUC-Sleep Cohort I", "recordings": len(records), "retained_epochs": sum(r["retained_epochs"] for r in records),
               "class_counts": total, "channels": CHANNELS, "fs": FS, "epoch_seconds": EPOCH_SECONDS,
               "reference_labels": "scorer 1 (y); scorer 2 in y_scorer2 (-1 where unavailable)",
               "cropping": f"{WAKE_MARGIN_EPOCHS} epochs (30 min) before the first and after the last scorer-1 sleep epoch",
               "montages": {m: sum(r["montage"] == m for r in records) for m in sorted({r["montage"] for r in records})},
               "excluded_outside_crop": sum(r["excluded_outside_crop"] for r in records),
               "mean_scorer_agreement": round(float(np.nanmean([r["scorer_agreement"] for r in records])), 4)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
