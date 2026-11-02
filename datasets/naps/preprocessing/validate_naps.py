"""Check preprocessed NAPS NPZs against records.csv and write per-recording signal QC (signal_qc.csv,
validation.json). Integrity checks are asserted; QC values are reported as robust z-score flags only."""
import sys
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
DATA = repo_paths.PREPROCESSED["naps"]
STAGES = ("W", "N1", "N2", "N3", "REM")


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    summary = json.loads((DATA / "summary.json").read_text())
    with (DATA / "records.csv").open() as stream:
        records = list(csv.DictReader(stream))
    channels = summary["channels"]
    total, qc = np.zeros(5, int), []
    for rec in records:
        path = DATA / rec["file"]
        assert file_hash(path) == rec["output_sha256"], rec["subject"]
        with np.load(path, allow_pickle=False) as z:
            x, y, idx, present = z["x"], z["y"], z["epoch_index"], z["channel_present"]
            assert x.shape == (len(y), len(channels), 3000) and x.dtype == np.float32
            assert list(z["channels"]) == channels and str(z["subject"]) == rec["subject"]
            assert np.isfinite(x[:, present]).all() and np.isnan(x[:, ~present]).all()
            assert set(np.unique(y)) <= set(range(5)) and len(y) == int(rec["retained_epochs"])
            assert np.all(np.diff(idx) > 0)
            assert np.array_equal(z["segment_id"], np.cumsum(np.r_[0, np.diff(idx) != 1]))
            assert np.all(z["onset_seconds"] >= 0) and np.all(z["onset_seconds"] + 30 <= float(rec["recording_seconds"]))
            counts = np.bincount(y, minlength=5)
            assert all(counts[k] == int(rec[s]) for k, s in enumerate(STAGES))
            total += counts
            f, p = welch(x[:, present], fs=100, nperseg=400, axis=-1)
            band = lambda lo, hi: p[..., (f >= lo) & (f < hi)].sum(-1)
            rel_delta = band(0.5, 4) / band(0.5, 30)
            nrem = np.isin(y, (2, 3))
            for k, c in enumerate(np.flatnonzero(present)):
                qc.append({"subject": rec["subject"], "channel": channels[c], "source": str(z["source_channels"][c]),
                           "native_fs": int(z["native_fs"][c]), "median_epoch_std_uV": float(np.median(x[:, c].std(-1))),
                           "nrem_rel_delta": float(np.median(rel_delta[nrem, k])) if nrem.any() else float("nan"),
                           "epochs_clip_over_1pct": int(np.sum(z["clip_fraction"][:, c] > 0.01)),
                           "epochs_flat_below_0p5uV": int(np.sum(x[:, c].std(-1) < 0.5))})
    assert total.sum() == summary["retained_epochs"]
    for name in channels:
        rows = [r for r in qc if r["channel"] == name]
        for key in ("median_epoch_std_uV", "nrem_rel_delta"):
            v = np.array([r[key] for r in rows])
            ok = np.isfinite(v)
            med = np.median(v[ok]) if ok.any() else 0
            mad = np.median(np.abs(v[ok] - med)) * 1.4826 if ok.any() else 0
            for r, value in zip(rows, v):
                r[f"{key}_robust_z"] = float((value - med) / mad) if mad > 0 and np.isfinite(value) else 0.0
    for r in qc:
        r["flag"] = ";".join(k.replace("_robust_z", "") for k in r if k.endswith("_robust_z") and abs(r[k]) > 4)
    with (DATA / "signal_qc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(qc[0]))
        writer.writeheader(); writer.writerows(qc)
    flagged = [(r["subject"], r["channel"], r["flag"], round(r["median_epoch_std_uV"], 1)) for r in qc if r["flag"]]
    result = {"recordings_checked": len(records), "epochs": int(total.sum()),
              "class_counts": {s: int(n) for s, n in zip(STAGES, total)}, "integrity_checks": "passed",
              "qc_flags_abs_robust_z_over_4": flagged}
    (DATA / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("recordings_checked", "epochs", "class_counts", "integrity_checks")}, indent=2))
    print(f"{len(flagged)} channel QC flags:")
    for f in flagged:
        print("  ", f)


if __name__ == "__main__":
    main()
