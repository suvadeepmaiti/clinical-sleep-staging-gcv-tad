# NAPS preprocessing

NSRR NAPS (North American Prodromal Synucleinopathy Consortium, https://sleepdata.org/datasets/naps):
138 NAPS1 participants with polysomnogram-confirmed idiopathic REM sleep behavior disorder (iRBD);
retrospective clinical PSGs from several sites and recording systems (Natus, Compumedics, Nihon
Kohden, Philips), scored centrally at the Mayo Clinic PSG Core (AASM). One visit per participant
(`nsrrid` "File N"; age 35–79, median 64; 110 male, 26 female, 2 unknown). Clinical study types: 85
diagnostic, 30 on CPAP, 3 on a mandibular advancement device, 18 split-night, 2 unknown.
Raw data (read only): `data/raw/naps` (`original/psg`, `harmonized`, `datasets`).

## Commands

```bash
cd datasets/naps/preprocessing
python check_alignment_naps.py --all --harmonized-only --csv alignment_all.csv   # timing diagnostics
python preprocess_naps.py --workers 2                                            # -> data/preprocessed/naps/
python validate_naps.py                                                          # integrity + signal_qc.csv
```

## Output (`data/preprocessed/naps/`)

One NPZ per recording, in the same format and channel order as `data/preprocessed/isleeps`, so the
iSLEEPS teachers and students can be applied directly. Plus `records.csv`, `summary.json`,
`signal_qc.csv`, `validation.json`.

| Key | Content |
|---|---|
| `x` | float32 `(N, 9, 3000)`, calibrated µV, 100 Hz, 30 s epochs; NaN for a missing channel |
| `y` | int64: 0=W, 1=N1, 2=N2, 3=N3, 4=REM |
| `channels` | `C3-M2, C4-M1, O1-M2, O2-M1, F3-M2, F4-M1, E1, E2, EMG` |
| `channel_present`, `source_channels`, `native_fs` | per channel; harmonized source labels `C3_M2 … LOC, ROC, EMG` |
| `epoch_index`, `onset_seconds`, `segment_id` | epoch identity, onset in seconds from EDF start, continuity runs |
| `in_wake_window_30min`, `clip_fraction` | optional Sleep-EDF wake rule; fraction of native samples at the digital limits |
| `study_type`, `age`, `sex`, `original_eog_label`, `original_emg_label` | metadata (EOG reference differs between recordings) |

## Dataset after preprocessing

138 recordings, **128,806 epochs**:

| W | N1 | N2 | N3 | REM |
|---:|---:|---:|---:|---:|
| 37,114 | 12,359 | 54,422 | 7,076 | 17,835 |

Excluded: 531 unscored (`?`), 53 movement (`M`), 697 outside the recording. Epochs per recording:
55–1,242 (median 978).

Channel availability: C4-M1 in 114 recordings, C3-M2 in 79, E1/E2 (LOC/ROC) in 137, chin EMG in 135.
All four SleepGPT-teacher channels (C3, C4, EMG, EOG) are present in 57 recordings.

| Channels present | Recordings |
|---|---:|
| all 9 | 57 |
| C4, E1, E2, EMG | 29 |
| C3, E1, E2, EMG (no C4) | 20 |
| C4, O2, F4, E1, E2, EMG | 20 |
| C4, O1, F3, E1, E2, EMG | 6 |
| E1, E2, EMG only | 2 |

## Processing rules

1. **Sources**: NSRR-harmonized EDF and `.annot` (class/Start/Stop in seconds). The harmonized stage
   sequences equal the original scorer files (`original/psg/File_N.txt`) in 136/138 recordings.
2. **Labels**: W, N1, N2, N3, R (NSRR merged S3/S4); `?` and `M` excluded. Epoch k covers
   [Start_k, Stop_k); epochs not fully inside the EDF are dropped.
3. **Channels**: harmonized `C3_M2, C4_M1, O1_M2, O2_M1, F3_M2, F4_M1, LOC, ROC, EMG` → the iSLEEPS
   order; missing → NaN. Every channel resampled once from its native rate (200/256/500/512 Hz) to
   100 Hz with `scipy.signal.resample_poly`. No other filtering, no normalization, no wake trimming.

## Annotation timing and data issues

`check_alignment_naps.py` measures, per recording, within-recording LDA staging accuracy of 1 s band
powers (EEG, LOC, EMG) averaged over 30 s windows shifted by −60…+60 s (`alignment_all.csv`).
Across the 138 recordings the mean accuracy peaks at 0 s (−2.0 points at −30 s, −2.8 points at
+30 s), so the NSRR timing is kept. EDF start times do not predict a clock-boundary offset (unlike
iSLEEPS).

| Recording | Issue | Handling |
|---|---|---|
| File_1 | Harmonized annotation = original scoring plus 2 leading W epochs; signal fits best at −60 s (0.807 vs 0.769) | onsets shifted −60 s |
| File_76 | Harmonized annotation holds two concatenated scorings (1201 + 970 rows); first block agrees with the original scorer file in 32% of epochs and fits the signal poorly (0.63) | labels from the original scorer file (fits at 0 s, 0.826) |
| File_34 | EDF holds 10,483 s; scoring covers 27,270 s (truncated recording); EOG ~145 µV | first 349 epochs kept; EOG to be checked in QC |
| File_30, 50, 51, 56, 68, 75, 80 | Shifted windows beat 0 s by more than 5 accuracy points | timing unchanged; `alignment_flag = possible_misalignment` in `records.csv` for sensitivity analyses |

Labels were never shifted on the classifier evidence alone.

## Validation and signal QC

`validate_naps.py`: all integrity checks passed (138 recordings, 128,806 epochs; hashes, shapes,
labels, epoch order, segments, counts, NaN only in missing channels). `signal_qc.csv` gives per
channel the median epoch SD, NREM delta share, clipped and flat epochs, and robust z-scores.

| Channel | Recordings | Median epoch SD, µV (p10–p90) | NREM delta share |
|---|---:|---|---:|
| C4-M1 | 114 | 13.6 (9.4–18.9) | 0.71 |
| C3-M2 | 79 | 13.0 (9.3–21.1) | 0.73 |
| E1 | 137 | 17.3 (10.8–28.9) | 0.83 |
| EMG | 135 | 5.5 (1.9–20.7) | 0.03 |

74 channel flags (|robust z| > 4), mostly chin EMG (51: amplitude 20–320 µV or unusual spectra,
consistent with varying EMG derivations and filters across recording systems). EEG/EOG flags:
C4 File_130 (low NREM delta); E1 amplitude File_34 (145 µV), File_43, File_45, File_106, File_107;
File_97 and File_106 have several high-amplitude channels. Flags are for sensitivity analyses; no
recording is excluded.

## Using NAPS with the iSLEEPS models

- Different population: idiopathic RBD (prodromal synucleinopathy; REM without atonia affects chin
  EMG in REM), some nights on CPAP/MAD, not stroke.
- EOG reference varies. The NAPS minimum montage recommends LOC-M2 / ROC-M1, but NAPS1 PSGs are
  retrospective clinical recordings; original labels: `EOG_LOC_M1` 43, `_Ref` 15, `_Fpz` 11, `_A2` 3,
  `_Fp2` 2, `_Fz` 1, unknown 63 (iSLEEPS: `E1-M2` throughout). Amplitude and polarity of `E1` can
  differ, which matters for EOG-based students.
- The 4-channel SleepGPT teacher needs C3, C4, EMG and EOG (57 recordings); the C4 and C4+EOG students
  can run on 114 recordings; the EOG-only student on 137.
