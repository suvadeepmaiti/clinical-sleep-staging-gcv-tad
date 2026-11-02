"""Check preprocessed iSLEEPS NPZs against records.csv and write per-recording signal QC."""
import sys
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
STAGES = ("W", "N1", "N2", "N3", "REM")
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_paths.PREPROCESSED["isleeps"])
    args = parser.parse_args()
    summary = json.loads((args.data_dir / "summary.json").read_text())
    with (args.data_dir / "records.csv").open() as stream:
        records = {r["subject"]: r for r in csv.DictReader(stream)}
    channels = summary["channels"]
    checks, qc, total = [], [], np.zeros(5, dtype=int)
    for subject, rec in records.items():
        path = args.data_dir / rec["file"]
        assert file_hash(path) == rec["output_sha256"], f"{subject}: hash mismatch"
        z = np.load(path)
        x, y, idx = z["x"], z["y"], z["epoch_index"]
        present = z["channel_present"]
        assert x.shape == (len(y), len(channels), 3000) and x.dtype == np.float32
        assert list(z["channels"]) == channels and str(z["subject"]) == subject
        assert np.isfinite(x[:, present]).all() and np.isnan(x[:, ~present]).all()
        assert y.dtype == np.int64 and set(np.unique(y)) <= set(range(5))
        assert np.all(np.diff(idx) > 0), f"{subject}: epoch_index not increasing"
        offset = int(z["hypnogram_offset_seconds"])
        assert np.array_equal(z["onset_seconds"], offset + 30 * idx) and z["onset_seconds"][0] >= 0
        assert np.array_equal(z["segment_id"], np.cumsum(np.r_[0, np.diff(idx) != 1]))
        assert len(y) == int(rec["retained_epochs"])
        counts = np.bincount(y, minlength=5)
        assert all(counts[i] == int(rec[s]) for i, s in enumerate(STAGES))
        total += counts
        # Signal QC: amplitude and N2/N3 spectral shape per channel (Welch over each epoch).
        f, p = welch(x[:, present], fs=100, nperseg=400, axis=-1)
        rel = {b: p[..., (f >= lo) & (f < hi)].sum(-1) / p[..., (f >= 0.5) & (f < 30)].sum(-1)
               for b, (lo, hi) in BANDS.items()}
        nrem = np.isin(y, (2, 3))
        for c, name in zip(np.flatnonzero(present), np.array(channels)[present]):
            k = list(np.flatnonzero(present)).index(c)
            qc.append({
                "subject": subject, "channel": name, "source": str(z["source_channels"][c]),
                "native_fs": int(z["native_fs"][c]),
                "median_epoch_std_uV": float(np.median(x[:, c].std(-1))),
                "nrem_rel_delta": float(np.median(rel["delta"][nrem, k])),
                "wake_rel_alpha": float(np.median(rel["alpha"][y == 0, k])) if np.any(y == 0) else np.nan,
                "epochs_clip_over_1pct": int(np.sum(z["clip_fraction"][:, c] > 0.01)),
                "epochs_flat_below_0p5uV": int(np.sum(x[:, c].std(-1) < 0.5)),
            })
        checks.append(subject)
    assert total.sum() == summary["retained_epochs"]
    assert {s: int(n) for s, n in zip(STAGES, total)} == summary["class_counts"]
    # Flag recordings whose channel QC is far from the cohort (robust z-score per channel).
    for name in channels:
        rows = [r for r in qc if r["channel"] == name]
        for key in ("median_epoch_std_uV", "nrem_rel_delta"):
            v = np.array([r[key] for r in rows])
            med, mad = np.median(v), np.median(np.abs(v - np.median(v))) * 1.4826
            for r, value in zip(rows, v):
                r[f"{key}_robust_z"] = float((value - med) / mad) if mad > 0 else 0.0
    for r in qc:
        r["flag"] = ";".join(k.replace("_robust_z", "") for k in r if k.endswith("_robust_z") and abs(r[k]) > 4)
    with (args.data_dir / "signal_qc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(qc[0]))
        writer.writeheader()
        writer.writerows(qc)
    flagged = [(r["subject"], r["channel"], r["flag"]) for r in qc if r["flag"]]
    result = {"recordings_checked": len(checks), "epochs": int(total.sum()),
              "class_counts": {s: int(n) for s, n in zip(STAGES, total)},
              "integrity_checks": "passed", "qc_flags_abs_robust_z_over_4": flagged}
    (args.data_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
