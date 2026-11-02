"""Prepare NSRR NAPS PSG (138 overnight recordings) for five-stage sleep staging, in the same NPZ
format as iSLEEPS/preprocessed so the same teachers and students can be applied.

Sources (read only): the NSRR NAPS download (repo_paths.RAW['naps'])
- harmonized/File_N.edf + harmonized/File_N.annot (NSRR-harmonized channel names; tab-separated
  class/Start/Stop in seconds from the harmonized EDF start). Stage sequences equal the original
  scorer files (original/psg/File_N.txt) in 136/138 recordings; see EXCEPTIONS for the others.
- datasets/naps-dataset-0.1.0.csv: age, sex, study type (1 diagnostic, 2 CPAP, 3 MAD, 4 split night).

Per recording (one per participant; nsrrid 'File N'):
- Labels W, N1, N2, N3, R -> 0..4 (NSRR already merged S3/S4 into N3); '?' and 'M' excluded.
- Epoch k covers [Start_k, Stop_k) of the annotation; epochs not fully inside the EDF are dropped.
- Channels in the iSLEEPS order C3-M2, C4-M1, O1-M2, O2-M1, F3-M2, F4-M1, E1, E2, EMG from the
  harmonized labels C3_M2, C4_M1, O1_M2, O2_M1, F3_M2, F4_M1, LOC, ROC, EMG; missing -> NaN with
  channel_present False. The original EOG/EMG labels (e.g. EOG_LOC_M1 vs EOG_LOC_Fpz) are recorded.
- Each channel resampled once from its native rate (200/256/500/512 Hz) to 100 Hz with
  scipy.signal.resample_poly; calibrated microvolts; no other filtering or normalization.
- No wake trimming; in_wake_window_30min marks the Sleep-EDF rule.
"""
import sys
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
RAW = repo_paths.RAW["naps"]
from naps_edf import read_edf, signal  # noqa: E402
CHANNELS = ("C3-M2", "C4-M1", "O1-M2", "O2-M1", "F3-M2", "F4-M1", "E1", "E2", "EMG")
SOURCES = {"C3-M2": "C3_M2", "C4-M1": "C4_M1", "O1-M2": "O1_M2", "O2-M1": "O2_M1", "F3-M2": "F3_M2",
           "F4-M1": "F4_M1", "E1": "LOC", "E2": "ROC", "EMG": "EMG"}
STAGES = ("W", "N1", "N2", "N3", "REM")
LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "?": -1, "M": -2}
FS, EPOCH_SECONDS = 100, 30
SAMPLES = FS * EPOCH_SECONDS
UV = {"uV", "µV", "uv"}
# From check_alignment_naps.py and comparison with the original scorer files (see PREPROCESSING.md).
EXCEPTIONS = {
    # Harmonized annotation = original scoring with 2 extra leading W epochs; the signal fits best
    # with onsets shifted by -60 s (LDA staging accuracy 0.807 vs 0.769 at 0 s).
    "File_1": {"onset_shift_seconds": -60},
    # Harmonized annotation holds two concatenated scorings (1201 + 970 rows); its first block
    # disagrees with the original scorer file (32% agreement) and fits the signal poorly (0.63).
    # The original scorer file (1201 epochs from the EDF start) fits well (0.826 at 0 s).
    "File_76": {"labels_from_original_txt": True},
    # File_34: EDF holds 10483 s while the scoring covers 27270 s (truncated recording); only epochs
    # inside the EDF are kept (handled by the general inside-recording rule).
}
TXT_MAP = {"W": "W", "S1": "N1", "S2": "N2", "S3": "N3", "S4": "N3", "4": "N3", "REM": "R", "M": "M", "?": "?", "N/A": "?"}


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def original_labels(i):
    with open(RAW / "original" / "psg" / f"File_{i}.edf", "rb") as f:
        h = f.read(256); ns = int(h[252:256]); body = f.read(ns * 256)
    return [body[k * 16:(k + 1) * 16].decode("latin-1").strip() for k in range(ns)]


def process(args):
    i, out_dir, meta = args
    subject = f"File_{i}"
    edf = read_edf(RAW / "harmonized" / f"{subject}.edf")
    with open(RAW / "harmonized" / f"{subject}.annot") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))[1:]
    if EXCEPTIONS.get(subject, {}).get("labels_from_original_txt"):
        lines = (RAW / "original" / "psg" / f"{subject}.txt").read_text().splitlines()[3:]
        txt = [TXT_MAP[p[4]] for p in (l.split() for l in lines) if len(p) >= 5]
        rows = [[s, str(30 * k), str(30 * k + 30)] for k, s in enumerate(txt)]
    stages = [r[0] for r in rows]
    unknown = set(stages) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"{subject}: unexpected labels {unknown}")
    start = np.array([float(r[1]) for r in rows]); stop = np.array([float(r[2]) for r in rows])
    if not (np.allclose(stop - start, EPOCH_SECONDS) and np.allclose(start, np.round(start))):
        raise ValueError(f"{subject}: annotations are not whole-second 30 s epochs")
    labels = np.array([LABEL_MAP[s] for s in stages], dtype=np.int64)
    shift = EXCEPTIONS.get(subject, {}).get("onset_shift_seconds", 0)
    onsets = np.round(start).astype(int) + shift
    seconds = edf["n_records"] * edf["record_seconds"]
    inside = (onsets >= 0) & (onsets + EPOCH_SECONDS <= seconds)
    indices = np.flatnonzero(inside & (labels >= 0))
    sleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    margin = 30 * 60 // EPOCH_SECONDS
    in_window = (indices >= sleep[0] - margin) & (indices <= sleep[-1] + margin) if len(sleep) else np.zeros(len(indices), bool)

    x = np.full((len(indices), len(CHANNELS), SAMPLES), np.nan, dtype=np.float32)
    clip = np.full((len(indices), len(CHANNELS)), np.nan, dtype=np.float32)
    present = np.zeros(len(CHANNELS), bool)
    native_fs = np.zeros(len(CHANNELS))
    for c, name in enumerate(CHANNELS):
        src = SOURCES[name]
        if src not in edf["labels"]:
            continue
        k = edf["labels"].index(src)
        if edf["dims"][k] not in UV:
            raise ValueError(f"{subject}: {src} unit {edf['dims'][k]!r}")
        values, fs = signal(edf, k)
        fs_int = int(round(fs))
        if abs(fs - fs_int) > 1e-6:
            raise ValueError(f"{subject}: non-integer rate {fs} for {src}")
        ratio = Fraction(FS, fs_int)
        resampled = resample_poly(values, ratio.numerator, ratio.denominator)
        starts = onsets[indices] * FS
        x[:, c] = resampled[starts[:, None] + np.arange(SAMPLES)]
        lo, hi = edf["offsets"][k], edf["offsets"][k + 1]
        digital = edf["records"][:, lo:hi].reshape(-1)
        nat = digital[(onsets[indices] * fs_int)[:, None] + np.arange(EPOCH_SECONDS * fs_int)]
        clip[:, c] = np.mean((nat <= edf["dmin"][k]) | (nat >= edf["dmax"][k]), axis=1)
        present[c], native_fs[c] = True, fs_int
    if not np.isfinite(x[:, present]).all():
        raise ValueError(f"{subject}: nonfinite samples")
    y = labels[indices]
    segment_id = np.cumsum(np.r_[0, np.diff(indices) != 1]).astype(np.int32)
    orig = original_labels(i)
    eog_ref = next((l for l in orig if l.startswith("EOG_LOC")), "")
    emg_src = next((l for l in orig if l.startswith(("EMG_Chin", "EMG_EMG1", "EMG_MENTON", "EMG_EMG2"))), "")
    path = out_dir / f"{subject}.npz"
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez(stream, x=x, y=y, subject=subject, recording=subject, channels=np.array(CHANNELS),
                 channel_present=present, source_channels=np.array([SOURCES[c] if p else "" for c, p in zip(CHANNELS, present)]),
                 native_fs=native_fs, units="uV", fs=FS, epoch_seconds=EPOCH_SECONDS,
                 epoch_index=indices, onset_seconds=onsets[indices], segment_id=segment_id,
                 in_wake_window_30min=in_window, clip_fraction=clip, edf_start_time=edf["start_time"],
                 study_type=meta.get("study_type", ""), age=meta.get("age", ""), sex=meta.get("sex", ""),
                 original_eog_label=eog_ref, original_emg_label=emg_src,
                 source_psg=f"harmonized/{subject}.edf", source_hypnogram=f"harmonized/{subject}.annot")
    temporary.replace(path)
    return {"file": path.name, "subject": subject, "study_type": meta.get("study_type", ""),
            "age": meta.get("age", ""), "sex": meta.get("sex", ""), "annotation_epochs": len(labels),
            "recording_seconds": seconds, "onset_shift_seconds": shift,
            "excluded_unscored": int(np.sum(inside & (labels == -1))), "excluded_movement": int(np.sum(inside & (labels == -2))),
            "excluded_outside_recording": int(np.sum(~inside)), "retained_epochs": len(y),
            "continuous_segments": int(segment_id[-1]) + 1 if len(y) else 0,
            **{s: int(np.sum(y == k)) for k, s in enumerate(STAGES)},
            "channels_present": ";".join(c for c, p in zip(CHANNELS, present) if p),
            "missing_channels": ";".join(c for c, p in zip(CHANNELS, present) if not p),
            "eeg_native_fs": int(native_fs[1] or native_fs[0]), "original_eog_label": eog_ref, "original_emg_label": emg_src,
            "outside_wake_window_30min": int(np.sum(~in_window)),
            "epochs_c4_clip_over_1pct": int(np.sum(np.nan_to_num(clip[:, 1]) > 0.01)),
            "psg_sha256": edf["sha256"], "annot_sha256": file_hash(RAW / "harmonized" / f"{subject}.annot"),
            "output_sha256": file_hash(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=repo_paths.PREPROCESSED["naps"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--files", type=int, nargs="+", default=list(range(1, 139)))
    args = parser.parse_args()
    meta = {}
    with (RAW / "datasets" / "naps-dataset-0.1.0.csv").open() as stream:
        for r in csv.DictReader(stream):
            meta[r["file_id"].replace(" ", "_")] = {"study_type": r["study_type"], "age": r["age"], "sex": r["sex"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(p.suffix == ".npz" for p in args.out_dir.iterdir()) and args.files == list(range(1, 139)):
        raise FileExistsError(f"Output folder already has NPZ files: {args.out_dir}")
    jobs = [(i, args.out_dir, meta.get(f"File_{i}", {})) for i in args.files if EXCEPTIONS.get(f"File_{i}", {}).get("exclude") is None]
    records = []
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for n, row in enumerate(pool.imap(process, jobs), 1):
            records.append(row)
            print(f"[{n:3d}/{len(jobs)}] {row['subject']}: {row['retained_epochs']} epochs, missing [{row['missing_channels']}]", flush=True)
    records.sort(key=lambda r: int(r["subject"].split("_")[1]))
    # Alignment diagnostics from check_alignment_naps.py (harmonized EDF, NSRR timing): best shift and
    # its LDA-accuracy gain over 0 s. Files gaining > 5 points are flagged as possibly misaligned; their
    # timing is NOT changed (only File_1 and File_76 have verified corrections).
    align_path = HERE / "alignment_all.csv"
    if align_path.exists():
        with align_path.open() as stream:
            align = {r["subject"]: r for r in csv.DictReader(stream)}
        deltas = list(range(-60, 61, 2))
        for r in records:
            a = align.get(r["subject"])
            if a is None:
                continue
            acc = {d: float(a[f"acc_{d:+d}"]) for d in deltas}
            best = max(acc, key=acc.get)
            r["alignment_best_shift_s"] = best
            r["alignment_gain_vs_0"] = round(acc[best] - acc[0], 4)
            r["alignment_flag"] = "possible_misalignment" if acc[best] - acc[0] > 0.05 and r["subject"] not in EXCEPTIONS else ""
    with (args.out_dir / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = {
        "dataset": "NSRR NAPS (138 overnight PSGs, sleep-apnea clinic cohort)", "source": str(RAW),
        "channels": CHANNELS, "channel_sources": SOURCES, "fs": FS, "epoch_seconds": EPOCH_SECONDS, "units": "uV",
        "label_names": STAGES, "label_map": LABEL_MAP, "excluded_labels": ["?", "M", "outside recording"],
        "exceptions": EXCEPTIONS, "recordings": len(records), "retained_epochs": sum(r["retained_epochs"] for r in records),
        "class_counts": {s: sum(r[s] for r in records) for s in STAGES},
        "recordings_with_c4": sum("C4-M1" in r["channels_present"].split(";") for r in records),
        "flagged_possible_misalignment": [r["subject"] for r in records if r.get("alignment_flag")],
        "recordings_with_c4_e1_emg_c3": sum(all(c in r["channels_present"].split(";") for c in ("C3-M2", "C4-M1", "EMG", "E1")) for r in records),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "script_sha256": file_hash(Path(__file__))}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("recordings", "retained_epochs", "class_counts", "recordings_with_c4",
                                              "recordings_with_c4_e1_emg_c3")}, indent=2))


if __name__ == "__main__":
    main()
