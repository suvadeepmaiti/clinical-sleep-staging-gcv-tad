"""Prepare iSLEEPS (ischemic-stroke PSG) EEG/EOG/chin EMG for five-stage training.

Corrections relative to github.com/suvadeepmaiti/iSLEEPS (commit 02391ac):
hypnogram epochs are placed using the xlsx clock time instead of assuming they
start at the EDF start, channel order and names are fixed across montages, each
channel is resampled once from its native rate, and the duplicate SN28 is skipped.
"""
import sys
import argparse
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import scipy
from scipy.signal import resample_poly

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
CHANNELS = ("C3-M2", "C4-M1", "O1-M2", "O2-M1", "F3-M2", "F4-M1", "E1-M2", "E2-M2", "EMG")
# Accepted EDF labels per canonical channel, in order of preference. A1/A2 files use
# auricular references and EOG1/EOG2 labels; SN13/SN17 have Chin 1/Chin 2 instead of EMG.
SOURCES = {
    "C3-M2": ("C3:M2", "C3:A2"), "C4-M1": ("C4:M1", "C4:A1"),
    "O1-M2": ("O1:M2", "O1:A2"), "O2-M1": ("O2:M1", "O2:A1"),
    "F3-M2": ("F3:M2", "F3:A2"), "F4-M1": ("F4:M1", "F4:A1"),
    "E1-M2": ("E1:M2", "EOG1:A2"), "E2-M2": ("E2:M2", "EOG2:A2"),
    "EMG": ("EMG", "Chin 1"),
}
STAGES = ("W", "N1", "N2", "N3", "REM")
LABEL_MAP = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 3, "REM": 4,
             "A": -1, "Artefact": -1, "Movement": -2}
DUPLICATES = {"SN28": "SN15"}  # Identical signals and hypnogram; only the EDF patient field differs.
FS = 100
EPOCH_SECONDS = 30
SAMPLES = FS * EPOCH_SECONDS
WAKE_MINUTES = 30
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
           "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def read_edf(path):
    """Read an EDF+C file into header fields and an int16 (records, samples) array."""
    raw = np.fromfile(path, dtype=np.uint8)
    h = raw[:256].tobytes().decode("latin-1")
    ns = int(h[252:256])
    body = raw[256:256 * (ns + 1)].tobytes().decode("latin-1")
    field = lambda offset, width: [body[offset * ns + i * width: offset * ns + (i + 1) * width].strip()
                                   for i in range(ns)]
    labels, dims = field(0, 16), field(96, 8)
    pmin, pmax, dmin, dmax = (np.array(field(o, 8), dtype=float) for o in (104, 112, 120, 128))
    nsamp = np.array(field(216, 8), dtype=int)
    n_records, record_seconds = int(h[236:244]), float(h[244:252])
    if not h[192:236].strip().startswith("EDF+C") or record_seconds != 1:
        raise ValueError(f"Expected continuous EDF+ with 1 s records: {path}")
    data = raw[256 * (ns + 1):].view("<i2")
    if data.size != n_records * nsamp.sum():
        raise ValueError(f"Truncated EDF: {path}")
    return {
        "labels": labels, "dims": dims, "pmin": pmin, "pmax": pmax, "dmin": dmin, "dmax": dmax,
        "nsamp": nsamp, "n_records": n_records, "start_time": h[176:184],
        "offsets": np.r_[0, np.cumsum(nsamp)], "records": data.reshape(n_records, nsamp.sum()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def edf_signal(edf, i):
    """Return digital samples and the digital-to-physical scale/offset of signal i."""
    lo, hi = edf["offsets"][i], edf["offsets"][i + 1]
    digital = edf["records"][:, lo:hi].reshape(-1)
    gain = (edf["pmax"][i] - edf["pmin"][i]) / (edf["dmax"][i] - edf["dmin"][i])
    return digital, gain, edf["pmin"][i] - gain * edf["dmin"][i]


def edf_annotations(edf):
    """Parse EDF+ TALs: returns (record start stamps, [(onset, text), ...])."""
    digital, _, _ = edf_signal(edf, edf["labels"].index("EDF Annotations"))
    stamps, events = [], []
    for record in digital.reshape(edf["n_records"], -1):
        for tal in record.tobytes().split(b"\x14\x00"):
            parts = tal.strip(b"\x00").split(b"\x14")
            if not parts or not parts[0]:
                continue
            onset = float(parts[0].split(b"\x15")[0])
            texts = [t.decode("latin-1").strip() for t in parts[1:] if t.strip()]
            if texts:
                events += [(onset, t) for t in texts]
            else:
                stamps.append(onset)
    return np.array(stamps), events


def read_sheet(path, name="Sleep profile"):
    """Minimal xlsx reader: {row number: {column letter: cell text}} for one sheet."""
    with zipfile.ZipFile(path) as z:
        book = ET.fromstring(z.read("xl/workbook.xml"))
        rid = next(s.get(f"{{{XLSX_NS['r']}}}id") for s in book.find("m:sheets", XLSX_NS)
                   if s.get("name") == name)
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == rid).lstrip("/")
        shared = [] if "xl/sharedStrings.xml" not in z.namelist() else [
            "".join(t.text or "" for t in si.iter(f"{{{XLSX_NS['m']}}}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", XLSX_NS)]
        sheet = ET.fromstring(z.read(target if target.startswith("xl/") else "xl/" + target))
    rows = {}
    for row in sheet.iter(f"{{{XLSX_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", XLSX_NS):
            v = c.find("m:v", XLSX_NS)
            if c.get("t") == "s":
                value = shared[int(v.text)]
            elif c.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{{{XLSX_NS['m']}}}t"))
            else:
                value = None if v is None else v.text
            cells[re.match(r"[A-Z]+", c.get("r")).group()] = value
        rows[int(row.get("r"))] = cells
    return rows


def read_hypnogram(path, edf_start_time):
    """Return stage strings and the hypnogram start relative to the EDF start (seconds)."""
    rows = read_sheet(path)
    meta = {cells.get("A"): cells.get("B") for r, cells in rows.items() if r < 8}
    if meta.get("Rate [s]") != "30" or rows[8].get("A") != "Time" or rows[8].get("B") != "Value":
        raise ValueError(f"Unexpected Sleep profile layout: {path}")
    entries = [(float(c["A"]), c.get("B")) for r, c in sorted(rows.items()) if r > 8]
    start = float(meta["Start Time"])  # Excel serial days
    elapsed = np.array([(t - start) * 86400 for t, _ in entries])
    if not np.allclose(elapsed, EPOCH_SECONDS * np.arange(len(entries)), atol=0.01):
        raise ValueError(f"Hypnogram rows are not consecutive 30 s epochs: {path}")
    xlsx_tod = (start % 1) * 86400
    edf = dt.datetime.strptime(edf_start_time, "%H.%M.%S")
    offset = (xlsx_tod - (edf.hour * 3600 + edf.minute * 60 + edf.second) + 43200) % 86400 - 43200
    if abs(offset - round(offset)) > 0.01:
        raise ValueError(f"Hypnogram offset is not whole seconds: {offset}")
    return [stage for _, stage in entries], int(round(offset))


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_recording(args):
    edf_path, xlsx_path, out_dir = args
    subject = edf_path.stem
    edf = read_edf(edf_path)
    stamps, events = edf_annotations(edf)
    if stamps[0] != 0 or not np.allclose(np.diff(stamps), 1):
        raise ValueError(f"Discontinuous EDF+ record stamps: {subject}")
    stages, offset = read_hypnogram(xlsx_path, edf["start_time"])
    unknown = set(stages) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected stage labels in {subject}: {unknown}")
    labels = np.array([LABEL_MAP[s] for s in stages], dtype=np.int64)

    # Hypnogram epoch k covers [offset + 30k, offset + 30k + 30) s after the EDF start.
    onsets = offset + EPOCH_SECONDS * np.arange(len(labels))
    inside = (onsets >= 0) & (onsets + EPOCH_SECONDS <= edf["n_records"])
    sleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    margin = WAKE_MINUTES * 60 // EPOCH_SECONDS
    indices = np.flatnonzero(inside & (labels >= 0))
    in_window = (indices >= sleep[0] - margin) & (indices <= sleep[-1] + margin)

    x = np.full((len(indices), len(CHANNELS), SAMPLES), np.nan, dtype=np.float32)
    clip = np.full((len(indices), len(CHANNELS)), np.nan, dtype=np.float32)
    present = np.zeros(len(CHANNELS), dtype=bool)
    sources, native_fs = [""] * len(CHANNELS), np.zeros(len(CHANNELS))
    for c, name in enumerate(CHANNELS):
        i = next((edf["labels"].index(s) for s in SOURCES[name] if s in edf["labels"]), None)
        if i is None:
            continue
        if edf["dims"][i] not in ("uV", "µV"):
            raise ValueError(f"Expected microvolts: {subject} {edf['labels'][i]}")
        fs = int(edf["nsamp"][i])
        digital, gain, shift = edf_signal(edf, i)
        ratio = Fraction(FS, fs)
        signal = resample_poly(digital * gain + shift, ratio.numerator, ratio.denominator)
        starts = onsets[indices] * FS
        x[:, c] = signal[starts[:, None] + np.arange(SAMPLES)]
        native = digital[(onsets[indices] * fs)[:, None] + np.arange(EPOCH_SECONDS * fs)]
        clip[:, c] = np.mean((native <= edf["dmin"][i]) | (native >= edf["dmax"][i]), axis=1)
        present[c], sources[c], native_fs[c] = True, edf["labels"][i], fs
    if not np.isfinite(x[:, present]).all():
        raise ValueError(f"Nonfinite samples in {subject}")
    y = labels[indices]
    segment_id = np.cumsum(np.r_[0, np.diff(indices) != 1]).astype(np.int32)
    lights_off = [t for t, e in events if e.lower() == "lights off"]
    lights_on = [t for t, e in events if e.lower() == "lights on"]

    path = out_dir / f"{subject}.npz"
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez(
            stream, x=x, y=y, subject=subject, recording=subject,
            channels=np.array(CHANNELS), channel_present=present, source_channels=np.array(sources),
            native_fs=native_fs, units="uV", fs=FS, epoch_seconds=EPOCH_SECONDS,
            epoch_index=indices, onset_seconds=onsets[indices], segment_id=segment_id,
            hypnogram_offset_seconds=offset, edf_start_time=edf["start_time"],
            first_sleep_epoch=sleep[0], last_sleep_epoch=sleep[-1],
            in_wake_window_30min=in_window, clip_fraction=clip,
            lights_off_seconds=np.array(lights_off), lights_on_seconds=np.array(lights_on),
            source_psg=edf_path.name, source_hypnogram=xlsx_path.name,
        )
    temporary.replace(path)
    eeg = present[:6]
    return {
        "file": path.name, "subject": subject, "hypnogram_epochs": len(labels),
        "recording_seconds": edf["n_records"], "hypnogram_offset_seconds": offset,
        "montage": "auricular (A1/A2 labels)" if any(":A" in s for s in sources) else "mastoid (M1/M2 labels)",
        "channels_present": ";".join(n for n, p in zip(CHANNELS, present) if p),
        "missing_channels": ";".join(n for n, p in zip(CHANNELS, present) if not p),
        "eeg_native_fs": int(native_fs[1]), "emg_source": sources[-1], "emg_native_fs": int(native_fs[-1]),
        "first_sleep_epoch": int(sleep[0]), "last_sleep_epoch": int(sleep[-1]),
        "excluded_artefact": int(np.sum(inside & (labels == -1))),
        "excluded_movement": int(np.sum(inside & (labels == -2))),
        "excluded_outside_recording": int(np.sum(~inside)),
        "retained_epochs": len(y), "outside_wake_window_30min": int(np.sum(~in_window)),
        "continuous_segments": int(segment_id[-1]) + 1,
        **{stage: int(np.sum(y == i)) for i, stage in enumerate(STAGES)},
        "epochs_any_eeg_eog_clip_over_1pct": int(np.sum(np.nanmax(clip[:, :8], axis=1) > 0.01)),
        "lights_off_seconds": ";".join(f"{t:.2f}" for t in lights_off),
        "lights_on_seconds": ";".join(f"{t:.2f}" for t in lights_on),
        "psg_sha256": edf["sha256"], "hypnogram_sha256": file_hash(xlsx_path),
        "output_sha256": file_hash(path), "all_eeg_present": bool(eeg.all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=repo_paths.RAW["isleeps"])
    parser.add_argument("--out-dir", type=Path, default=repo_paths.PREPROCESSED["isleeps"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    edfs = sorted(args.raw_dir.glob("SN*.edf"), key=lambda p: int(p.stem[2:]))
    if [p.stem for p in edfs] != [f"SN{i}" for i in range(1, 101)]:
        raise ValueError("Expected SN1.edf ... SN100.edf")
    jobs = []
    for edf in edfs:
        xlsx = args.raw_dir / f"{edf.stem}.xlsx"
        if not xlsx.exists():
            raise FileNotFoundError(xlsx)
        if edf.stem not in DUPLICATES:
            jobs.append((edf, xlsx, args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output folder must be empty: {args.out_dir}. Choose a new --out-dir.")
    records = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap(process_recording, jobs), 1):
            records.append(row)
            print(f"[{i:02d}/{len(jobs)}] {row['subject']}: {row['retained_epochs']} epochs, "
                  f"offset {row['hypnogram_offset_seconds']:+d} s, missing [{row['missing_channels']}]", flush=True)
    with (args.out_dir / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "dataset": "iSLEEPS (ischemic stroke PSG)",
        "source": "https://india-data.org/dataset-details/0b801dfa-4e42-4ec6-9c56-c6892b907ed2",
        "reference_code": "https://github.com/suvadeepmaiti/iSLEEPS (commit 02391ac35f623f4823db71126fcd9d8ebc1a582a)",
        "channels": CHANNELS, "channel_sources": SOURCES, "shape": "(epochs, channels, samples)",
        "missing_channel_fill": "NaN; see channel_present", "fs": FS, "epoch_seconds": EPOCH_SECONDS,
        "units": "uV", "x_dtype": "float32", "y_dtype": "int64", "label_names": STAGES,
        "label_map": LABEL_MAP, "excluded_labels": ["A/Artefact", "Movement", "partially outside EDF"],
        "epoch_timing": "Epoch k onset = hypnogram clock start - EDF clock start + 30k seconds",
        "wake_trimming": f"None applied; in_wake_window_30min marks the {WAKE_MINUTES}-minute Sleep-EDF rule",
        "filtering": "Only the anti-aliasing FIR in scipy.signal.resample_poly (native rate -> 100 Hz)",
        "saved_normalization": "None (calibrated microvolts)",
        "excluded_recordings": {k: f"byte-identical signals and hypnogram to {v}" for k, v in DUPLICATES.items()},
        "subjects": len(records), "retained_epochs": sum(r["retained_epochs"] for r in records),
        "class_counts": {s: sum(r[s] for r in records) for s in STAGES},
        "excluded": {k: sum(r[k] for r in records) for k in
                     ("excluded_artefact", "excluded_movement", "excluded_outside_recording")},
        "outside_wake_window_30min": sum(r["outside_wake_window_30min"] for r in records),
        "recordings_missing_channels": {r["subject"]: r["missing_channels"] for r in records if r["missing_channels"]},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "preprocessing_script_sha256": file_hash(Path(__file__)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("subjects", "retained_epochs", "class_counts", "excluded",
                                              "outside_wake_window_30min")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
