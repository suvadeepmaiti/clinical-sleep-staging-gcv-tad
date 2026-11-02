"""Empirical annotation timing check for NAPS: within-recording LDA staging accuracy of 1 s band
powers (EEG, LOC, EMG) averaged over 30 s windows shifted by delta seconds relative to the annotation
onsets, computed against both the harmonized and the original EDF of each recording."""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
from naps_edf import read_edf, signal  # noqa: E402

RAW = repo_paths.RAW["naps"]
MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 30), (30, 45)]


def features(edf, names):
    out = []
    if not names:
        return None
    for name in names:
        if name not in edf["labels"]:
            return None
        x, fs = signal(edf, edf["labels"].index(name))
        fs = int(round(fs)); n = len(x) // fs
        f, p = welch(x[:n * fs].reshape(n, fs), fs=fs, nperseg=fs, axis=-1)
        out.append(np.log(np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in BANDS], 1) + 1e-6))
    return np.concatenate(out, 1)


def curve(F, onsets, labels, deltas):
    cs = np.vstack([np.zeros(F.shape[1]), np.cumsum(F, 0)])
    keep = [k for k, l in enumerate(labels) if l in MAP and all(0 <= onsets[k] + d and onsets[k] + d + 30 <= len(F) for d in deltas)]
    y = np.array([MAP[labels[k]] for k in keep])
    acc = {}
    for d in deltas:
        s = (np.array([onsets[k] for k in keep]) + d).astype(int)
        acc[d] = cross_val_score(LinearDiscriminantAnalysis(), (cs[s + 30] - cs[s]) / 30, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
    return acc, len(keep)


def main(ids, harmonized_only=False, csv_path=None):
    deltas = list(range(-60, 61, 2))
    out_rows = []
    for i in ids:
        ann = list(csv.reader(open(RAW / "harmonized" / f"File_{i}.annot"), delimiter="\t"))[1:]
        labels, onsets = [a[0] for a in ann], [int(round(float(a[1]))) for a in ann]
        if i == 76:  # two concatenated scorings in the harmonized file; use the original scorer file
            M = {"W": "W", "S1": "N1", "S2": "N2", "S3": "N3", "S4": "N3", "4": "N3", "REM": "R", "M": "M", "?": "?", "N/A": "?"}
            labels = [M[p[4]] for p in (l.split() for l in (RAW / "original" / "psg" / "File_76.txt").read_text().splitlines()[3:]) if len(p) >= 5]
            onsets = [30 * k for k in range(len(labels))]
        sources = [("harmonized", RAW / "harmonized" / f"File_{i}.edf")]
        if not harmonized_only:
            sources.append(("original", RAW / "original" / "psg" / f"File_{i}.edf"))
        for kind, path in sources:
            edf = read_edf(path)
            eeg = next((c for c in (["C4_M1", "C3_M2"] if kind == "harmonized" else ["EEG_C4_M1", "EEG_C3_M2", "EEG_C3_M1"]) if c in edf["labels"]), None)
            eog = next((c for c in (["LOC"] if kind == "harmonized" else ["EOG_LOC_M1", "EOG_LOC_M2", "EOG_LOC_Fpz", "EOG_LOC"]) if c in edf["labels"]), None)
            emg = next((c for c in (["EMG"] if kind == "harmonized" else ["EMG_Chin_1_Ref", "EMG_EMG1", "EMG_MENTON", "EMG_EMG2"]) if c in edf["labels"]), None)
            F = features(edf, [c for c in (eeg, eog, emg) if c])
            dur = edf["n_records"] * edf["record_seconds"]
            del edf
            if F is None:
                print(f"File_{i} {kind}: channels missing"); continue
            acc, n = curve(F, onsets, labels, deltas)
            best = max(acc, key=acc.get)
            print(f"File_{i:3d} {kind:10s} dur={dur:7.0f}s epochs_used={n:4d} best_delta={best:+3d}s acc(best)={acc[best]:.4f} "
                  f"acc(0)={acc[0]:.4f} acc(-30)={acc[-30]:.4f} acc(+30)={acc[30]:.4f}", flush=True)
            out_rows.append({"subject": f"File_{i}", "edf": kind, "duration_s": dur, "epochs_used": n, "best_delta_s": best,
                             **{f"acc_{d:+d}": round(acc[d], 5) for d in deltas}})
            if csv_path:
                with open(csv_path, "w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(out_rows[0]))
                    writer.writeheader(); writer.writerows(out_rows)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("files", type=int, nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--harmonized-only", action="store_true")
    p.add_argument("--csv", type=Path)
    a = p.parse_args()
    main(list(range(1, 139)) if a.all else a.files, a.harmonized_only, a.csv)
