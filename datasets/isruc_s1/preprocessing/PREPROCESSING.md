# ISRUC-Sleep Cohort I preprocessing

ISRUC-Sleep Cohort I (subgroup I): 100 adults with sleep disorders, one overnight PSG each, recorded at the
Sleep Medicine Centre of the Hospital of Coimbra University and scored by two experts (AASM).
Source: https://sleeptight.isr.uc.pt (MEGA folder, 14.12 GB). Cite Khalighi et al., Comput. Methods
Programs Biomed. 124 (2016) 180-192. Raw data: `ISRUC-1/N/` (read only: `N.rec`, `N_1.txt`, `N_2.txt`).

## Commands

```bash
cd datasets/isruc_s1/preprocessing
python preprocess_isruc.py --workers 3     # -> data/preprocessed/isruc/
python validate_isruc.py                   # integrity + signal_qc.csv + validation.json
```

## Output (`data/preprocessed/isruc/`)

One NPZ per recording (`ISRUC1_NNN.npz`, NNN = ISRUC id), in the iSLEEPS/NAPS format, plus `records.csv`,
`summary.json`, `signal_qc.csv`, `validation.json`.

| Key | Content |
|---|---|
| `x` | float32 `(N, 9, 3000)`, µV as recorded, 100 Hz, 30 s epochs; NaN for a missing channel |
| `y` | int64 scorer-1 stages: 0=W, 1=N1, 2=N2, 3=N3, 4=REM (reference labels) |
| `y_scorer2` | scorer-2 stages for the same epochs (-1 where scorer 2 has no label) |
| `channels` | `C3-M2, C4-M1, O1-M2, O2-M1, F3-M2, F4-M1, E1-M2, E2-M1, EMG` |
| `channel_present`, `source_channels`, `native_fs`, `montage` | per channel / recording |
| `epoch_index`, `onset_seconds`, `segment_id` | epoch identity (index from EDF start), onset, continuity runs |
| `clip_fraction` | fraction of native samples at the digital limits, per epoch and channel |

## Dataset after preprocessing

100 recordings, **87,984 epochs** (299-1,062 per recording, median 884):

| W | N1 | N2 | N3 | REM |
|---:|---:|---:|---:|---:|
| 18,776 | 11,513 | 28,287 | 17,480 | 11,928 |

Scorer 1 vs scorer 2 on the retained epochs: accuracy 0.816, Cohen's κ 0.761 (per recording 0.62-0.94,
median 0.82), the human-agreement reference for model results.

## Processing rules

1. **Channels.** `-M1/-M2` derivations (82 recordings); `-A1/-A2` (17; A1/A2 are the mastoids, used as M1/M2);
   recording 40 is monopolar (`C3, A2, LOC, ...`), so derivations are computed (`C3 - A2`, `LOC - A2`, ...).
   EOG: `E1-M2` / `LOC-A2`, `E2-M1` / `ROC-A1`. Chin EMG: `X1`, or `24` in the numbered layout (identified by
   spectrum: ~90% power at 20-90 Hz; ECG is `X2`/`25`, leg EMG `X3`-`X4`/`26`-`27`). All 200 Hz, resampled
   once to 100 Hz (`resample_poly`); no filtering, no rescaling. Recording 8 has no F3/F4.
2. **Labels.** Values 0, 1, 2, 3, 5 -> W, N1, N2, N3, REM. Epoch k = [30k, 30k + 30) s from the EDF start;
   epochs beyond the EDF or the scorer-1 file are dropped (recording 18: scorer 2 lists 7 extra epochs).
3. **Cropping.** From 30 min (60 epochs) before the first to 30 min after the last scorer-1 sleep epoch,
   clipped to the recording: 2,203 epochs removed in 29 recordings (the others start or end within 30 min of
   sleep).

## Data issues

| Issue | Recordings | Handling |
|---|---|---|
| **EEG/EOG/EMG amplitude ~10x lower** in the A1/A2 and monopolar recordings (median epoch SD: C4 1.36 vs 15.2 µV, E1 1.38 vs 15.5 µV, EMG 0.36 vs 2.85 µV), with the same spectral shape (NREM delta share 0.74 vs 0.73); headers give physical ranges of ±25 vs ±2490-5013 | 1-10, 12, 15, 16, 18, 22, 24, 26, 40 | kept as recorded (`montage` field marks them); likely a gain/unit difference in the source files. Models with fixed input scaling (SleepGPT slots) need per-recording normalization or a gain correction chosen without test data |
| Two extra files whose names and EDF header contain a person's identifying details (the EDF signal data equal `65.rec`, differing only in header bytes 9-58) | 65 | never read; `65.rec` (de-identified) is used. Recommend deleting them and informing the dataset providers |
| High amplitude in all EEG/EOG channels (35-48 µV) | 25 | flagged (`signal_qc.csv`) |
| Low NREM delta share in C3/F3/F4 | 74 | flagged |
| Chin EMG amplitude or spectrum outliers | 21, 22, 29, 42, 63, 71 | flagged |

Flags (|robust z| > 4 within channel) are for sensitivity analyses; no recording is excluded.
