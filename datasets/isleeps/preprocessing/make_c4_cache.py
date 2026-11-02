"""Build C4 caches for the BIOT experiment from the corrected iSLEEPS preprocessing.

- data/c4_100hz/: the C4-M1 channel copied unchanged from iSLEEPS/preprocessed (student input).
- data/c4_200hz/: C4 re-extracted from the EDFs at 200 Hz (BIOT's pretraining rate) with the
  same reader, hypnogram alignment and epoch selection, resampled once from the native rate.
Epoch identities and labels are asserted identical between the two.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
import preprocess_isleeps as P  # noqa: E402

CHANNEL = "C4-M1"
FS_HIGH = 200


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(args):
    subject, raw_dir, source_dir, out_low, out_high = args
    with np.load(source_dir / f"{subject}.npz", allow_pickle=False) as d:
        c = d["channels"].tolist().index(CHANNEL)
        assert d["channel_present"][c]
        meta = {k: d[k] for k in ("y", "epoch_index", "onset_seconds", "segment_id", "subject",
                                  "hypnogram_offset_seconds")}
        low = np.ascontiguousarray(d["x"][:, c:c + 1, :])
        source = str(d["source_channels"][c])
    np.savez(out_low / f"{subject}.npz", x=low, channels=np.array([CHANNEL]), channel_present=np.array([True]),
             source_channels=np.array([source]), fs=100, units="uV", **meta)

    edf = P.read_edf(raw_dir / f"{subject}.edf")
    stages, offset = P.read_hypnogram(raw_dir / f"{subject}.xlsx", edf["start_time"])
    labels = np.array([P.LABEL_MAP[s] for s in stages], dtype=np.int64)
    onsets = offset + P.EPOCH_SECONDS * np.arange(len(labels))
    inside = (onsets >= 0) & (onsets + P.EPOCH_SECONDS <= edf["n_records"])
    indices = np.flatnonzero(inside & (labels >= 0))
    assert offset == int(meta["hypnogram_offset_seconds"])
    assert np.array_equal(indices, meta["epoch_index"]) and np.array_equal(labels[indices], meta["y"])
    i = edf["labels"].index(source)
    fs = int(edf["nsamp"][i])
    digital, gain, shift = P.edf_signal(edf, i)
    ratio = Fraction(FS_HIGH, fs)
    signal = resample_poly(digital * gain + shift, ratio.numerator, ratio.denominator)
    samples = P.EPOCH_SECONDS * FS_HIGH
    high = signal[(onsets[indices] * FS_HIGH)[:, None] + np.arange(samples)].astype(np.float32)
    assert np.isfinite(high).all()
    # Sanity: the 200 Hz epochs, decimated to 100 Hz, should closely match the stored 100 Hz epochs.
    corr = np.corrcoef(resample_poly(high[:50], 1, 2, axis=-1).ravel(), low[:50, 0].ravel())[0, 1]
    np.savez(out_high / f"{subject}.npz", x=high[:, None, :], channels=np.array([CHANNEL]),
             channel_present=np.array([True]), source_channels=np.array([source]), fs=FS_HIGH,
             native_fs=fs, units="uV", **meta)
    return {"subject": subject, "epochs": len(indices), "source_channel": source, "native_fs": fs,
            "corr_200hz_vs_100hz": round(float(corr), 6),
            "c4_100hz_sha256": file_hash(out_low / f"{subject}.npz"),
            "c4_200hz_sha256": file_hash(out_high / f"{subject}.npz")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=repo_paths.RAW["isleeps"])
    parser.add_argument("--source-dir", type=Path, default=repo_paths.PREPROCESSED["isleeps"])
    parser.add_argument("--out-dir", type=Path, default=repo_paths.ISLEEPS_CACHE)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out_low, out_high = args.out_dir / "c4_100hz", args.out_dir / "c4_200hz"
    for folder in (out_low, out_high):
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            raise FileExistsError(f"Output folder must be empty: {folder}")
    with (args.source_dir / "records.csv").open() as stream:
        subjects = [r["subject"] for r in csv.DictReader(stream)]
    with Pool(args.workers) as pool:
        rows = []
        for row in pool.imap(build, [(s, args.raw_dir, args.source_dir, out_low, out_high) for s in subjects]):
            rows.append(row)
            print(f"{row['subject']}: {row['epochs']} epochs, {row['source_channel']} {row['native_fs']} Hz, "
                  f"corr={row['corr_200hz_vs_100hz']}", flush=True)
    for folder in (out_low, out_high):
        shutil.copy2(args.source_dir / "records.csv", folder / "records.csv")
    with (args.out_dir / "c4_cache_records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"subjects": len(rows), "epochs": sum(r["epochs"] for r in rows), "channel": CHANNEL,
               "c4_100hz": "copied unchanged from iSLEEPS/preprocessed", "c4_200hz": "EDF -> resample_poly -> 200 Hz",
               "min_corr_200hz_vs_100hz": min(r["corr_200hz_vs_100hz"] for r in rows),
               "source_preprocessing_sha256": file_hash(HERE / "preprocess_isleeps.py"),
               "script_sha256": file_hash(Path(__file__))}
    (args.out_dir / "c4_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
