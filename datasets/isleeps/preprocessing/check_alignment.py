"""Empirically check hypnogram-to-EDF timing for iSLEEPS.

For each recording, 1 s log band powers (C4, E1, chin EMG) are averaged over 30 s
windows starting at (clock offset + 30k + delta) seconds, and within-recording 5-fold
LDA staging accuracy is computed for each delta. If the clock alignment used by
preprocess_isleeps.py is right, accuracy should peak near delta = 0; delta = -offset
is the alignment assumed by the reference repository (epoch k starts at 30k).
"""
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score

import preprocess_isleeps as P

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
DELTAS = np.arange(-30, 31, 2)
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 30), (30, 50)]


def recording_curve(args):
    edf_path, xlsx_path = args
    edf = P.read_edf(edf_path)
    stages, offset = P.read_hypnogram(xlsx_path, edf["start_time"])
    labels = np.array([P.LABEL_MAP[s] for s in stages])
    feats = []
    for name in ("C4-M1", "E1-M2", "EMG"):
        i = next(edf["labels"].index(s) for s in P.SOURCES[name] if s in edf["labels"])
        digital, gain, shift = P.edf_signal(edf, i)
        fs = int(edf["nsamp"][i])
        f, p = welch((digital * gain + shift).reshape(-1, fs), fs=fs, nperseg=fs, axis=-1)
        feats.append(np.log(np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in BANDS], 1) + 1e-6))
    cumulative = np.vstack([np.zeros(3 * len(BANDS)), np.cumsum(np.concatenate(feats, 1), 0)])
    seconds = edf["n_records"]
    candidates = sorted(set(DELTAS) | {-offset})
    # Use the same epochs for every delta so accuracies are comparable.
    keep = [k for k, label in enumerate(labels) if label >= 0 and
            all(0 <= offset + 30 * k + d and offset + 30 * k + d + 30 <= seconds for d in candidates)]
    y = labels[keep]
    accuracy = {}
    for d in candidates:
        starts = offset + 30 * np.array(keep) + d
        X = (cumulative[starts + 30] - cumulative[starts]) / 30
        accuracy[d] = cross_val_score(LinearDiscriminantAnalysis(), X, y,
                                      cv=KFold(5, shuffle=True, random_state=0)).mean()
    grid = {d: accuracy[d] for d in DELTAS}
    return {"subject": edf_path.stem, "offset_seconds": offset, "epochs_used": len(keep),
            "best_delta_seconds": int(max(grid, key=grid.get)), "accuracy_clock_delta0": accuracy[0],
            "reference_delta_seconds": -offset, "accuracy_reference_alignment": accuracy[-offset],
            "best_accuracy": max(grid.values()),
            **{f"acc_delta_{d:+d}": accuracy[d] for d in DELTAS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=repo_paths.RAW["isleeps"])
    parser.add_argument("--out", type=Path, default=repo_paths.PREPROCESSED["isleeps"] / "alignment_check.csv")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = [(args.raw_dir / f"SN{i}.edf", args.raw_dir / f"SN{i}.xlsx")
            for i in range(1, 101) if f"SN{i}" not in P.DUPLICATES]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(recording_curve, jobs))
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = np.array([r["best_delta_seconds"] for r in rows])
    gain = np.array([r["accuracy_clock_delta0"] - r["accuracy_reference_alignment"] for r in rows
                     if r["reference_delta_seconds"] != 0])
    mean_curve = {d: np.mean([r[f"acc_delta_{d:+d}"] for r in rows]) for d in DELTAS}
    print(f"recordings {len(rows)}; median best delta {np.median(best):+.0f} s "
          f"(IQR {np.percentile(best, 25):+.0f} to {np.percentile(best, 75):+.0f})")
    print(f"mean-curve peak delta {max(mean_curve, key=mean_curve.get):+d} s")
    print(f"clock minus reference-alignment accuracy, {len(gain)} recordings with nonzero offset: "
          f"mean {gain.mean():+.4f}, positive in {np.sum(gain > 0)}")


if __name__ == "__main__":
    main()
