"""Build data/psg4_100hz/: C3, C4, chin EMG and E1 EOG at 100 Hz for the multichannel teacher.

Channels are copied unchanged from iSLEEPS/preprocessed and stored in SleepGPT's pretraining slot
order (C3, C4, EMG, EOG), which is its own 'shhs' channel subset. All four are present in all 99
patients. Epoch identities and labels are asserted identical to data/c4_100hz (the student input).
"""
import sys
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
SOURCE = repo_paths.PREPROCESSED["isleeps"]
CHANNELS = ["C3-M2", "C4-M1", "EMG", "E1-M2"]  # SleepGPT slots 0 (C3), 1 (C4), 2 (EMG), 3 (EOG)


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    out = repo_paths.ISLEEPS_CACHE / "psg4_100hz"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output folder must be empty: {out}")
    with (SOURCE / "records.csv").open() as stream:
        subjects = [r["subject"] for r in csv.DictReader(stream)]
    rows = []
    for subject in subjects:
        with np.load(SOURCE / f"{subject}.npz", allow_pickle=False) as d, \
                np.load(repo_paths.ISLEEPS_CACHE / "c4_100hz" / f"{subject}.npz", allow_pickle=False) as c4:
            idx = [d["channels"].tolist().index(c) for c in CHANNELS]
            assert d["channel_present"][idx].all(), subject
            x = np.ascontiguousarray(d["x"][:, idx, :], dtype=np.float32)
            assert np.isfinite(x).all()
            assert np.array_equal(d["epoch_index"], c4["epoch_index"]) and np.array_equal(d["y"], c4["y"])
            assert np.array_equal(x[:, 1], c4["x"][:, 0])
            sources = d["source_channels"][idx]
            np.savez(out / f"{subject}.npz", x=x, channels=np.array(CHANNELS), channel_present=np.ones(4, bool),
                     source_channels=sources, native_fs=d["native_fs"][idx], fs=100, units="uV",
                     **{k: d[k] for k in ("y", "epoch_index", "onset_seconds", "segment_id", "subject",
                                          "hypnogram_offset_seconds")})
        rows.append({"subject": subject, "epochs": len(x), "sources": ";".join(sources),
                     "sha256": file_hash(out / f"{subject}.npz")})
    shutil.copy2(SOURCE / "records.csv", out / "records.csv")
    summary = {"subjects": len(rows), "epochs": sum(r["epochs"] for r in rows), "channels": CHANNELS,
               "sleepgpt_slots": ["C3", "C4", "EMG", "EOG"], "source": "iSLEEPS/preprocessed (unchanged values)",
               "script_sha256": file_hash(Path(__file__)), "files": rows}
    (repo_paths.ISLEEPS_CACHE / "psg4_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("subjects", "epochs", "channels")}))


if __name__ == "__main__":
    main()
