# iSLEEPS preprocessing

iSLEEPS: overnight PSG of ischemic-stroke patients (Maiti et al., *Scientific Data* 2026,
doi:10.1038/s41597-026-06747-w). Raw data: `data/raw/iSLEEPS` (`SN1`–`SN100`, one
`.edf` + one `.xlsx` hypnogram per patient, one night each). Raw files are read only.

## Commands

```bash
cd datasets/isleeps/preprocessing
python preprocess_isleeps.py      # -> data/preprocessed/isleeps/ (must be empty)
python validate_isleeps.py        # integrity checks + signal_qc.csv + validation.json
python check_alignment.py         # empirical hypnogram timing check -> alignment_check.csv
python make_c4_cache.py           # -> data/cache/isleeps/c4_100hz/
python make_psg4_cache.py         # -> data/cache/isleeps/psg4_100hz/ (teacher input)
```

## Output (`data/preprocessed/isleeps/`)

One uncompressed `SN*.npz` per patient (99 files, 9.4 GB), plus `records.csv`,
`summary.json`, `signal_qc.csv`, `validation.json`, `alignment_check.csv`.

| Key | Content |
|---|---|
| `x` | float32 `(N, 9, 3000)`, calibrated µV, 100 Hz, 30 s epochs; NaN for a missing channel |
| `y` | int64 `(N,)`: 0=W, 1=N1, 2=N2, 3=N3, 4=REM |
| `channels` | `C3-M2, C4-M1, O1-M2, O2-M1, F3-M2, F4-M1, E1-M2, E2-M2, EMG` (fixed order) |
| `channel_present`, `source_channels`, `native_fs` | per channel: available?, original EDF label, original rate |
| `epoch_index` | hypnogram row (original 30 s timeline); gaps = excluded epochs |
| `segment_id` | increments at each gap; sequences/transition pairs must not cross it |
| `onset_seconds` | epoch start relative to EDF start (= `hypnogram_offset_seconds + 30·epoch_index`) |
| `in_wake_window_30min` | True if within 30 min of first/last sleep (the Sleep-EDF rule); not applied by default |
| `clip_fraction` | `(N, 9)` fraction of native samples at the digital min/max |
| `lights_off_seconds`, `lights_on_seconds` | EDF+ markers (incomplete in some files; informational only) |

## Dataset after preprocessing

99 patients, **92,530 epochs**:

| W | N1 | N2 | N3 | REM |
|---:|---:|---:|---:|---:|
| 24,658 | 9,323 | 39,114 | 8,212 | 11,223 |

Excluded: 1,961 artefact epochs (`A`/`Artefact`; the first hypnogram row is always `A`),
133 epochs partially outside the EDF, and recording SN28 (see below). Applying the
optional 30-minute wake rule would remove a further 2,615 wake epochs.

## Processing rules

1. **Epoch timing.** The xlsx `Sleep profile` sheet gives a clock start (Excel serial) and
   consecutive 30 s rows. Its start is on a :00/:30 clock boundary, 0–29 s *before* the EDF
   start in 94 files and 193–266 s *after* it in SN24, SN30, SN36, SN38 and SN42 (scoring
   began near lights-off). Epoch k therefore starts at `offset + 30k` s from the EDF start,
   where `offset = xlsx clock start − EDF clock start` (whole seconds; EDF+ subsecond start
   is 0 in all files). Epochs not fully inside the recording are dropped.
   `check_alignment.py` (within-recording LDA on 1 s band powers of C4/E1/EMG, shifts of
   −30…+30 s) confirms this: across 99 recordings the best shift has median −2 s (IQR −8 to
   +1), and clock alignment beats the reference repository's alignment by +2.2 accuracy
   points on average (better in 76 of 91 recordings with a nonzero offset).
2. **Labels.** Wake→W, N1, N2, N3 (N4→N3; none present), REM. `A`/`Artefact`/`Movement`
   are excluded. Any other label stops the run.
3. **Channels.** The fixed order above. Files use either M1/M2 (64 files) or A1/A2 labels
   (36 files); `C4:A1`→`C4-M1`, `EOG1:A2`→`E1-M2`, etc., with the true label kept in
   `source_channels`. Chin EMG is `EMG`; SN13/SN17 have only `Chin 1`/`Chin 2`, and
   `Chin 1` is used. Frontal leads are missing in 13 files (NaN, `channel_present=False`).
4. **Resampling.** Each channel is resampled once from its native rate (128/256/512 Hz)
   to 100 Hz with `scipy.signal.resample_poly` (polyphase anti-aliasing FIR). No other
   filtering or normalization. EMG content above 50 Hz is lost at 100 Hz.
5. **Wake.** All scored epochs are kept (in-lab recordings, ~8 h). The Sleep-EDF 30-minute
   rule is available through `in_wake_window_30min`.

## Data issues found

- **SN28 duplicates SN15**: identical signal bytes and hypnogram; only the EDF patient
  field differs. SN28 is excluded to prevent subject leakage.
- EDF patient fields of SN11–SN20 read `SN1`–`SN10`: a renumbering artefact (different
  start times and data), not duplicates.
- SN1/SN2 use a different export (all signals 256 Hz, `HP:10Hz` in every header), but their
  EEG delta power is normal, so the header filter field is not taken at face value.
- `signal_qc.csv` flags 43 channel-recordings with |robust z| > 4 in amplitude or NREM delta
  share (mostly EMG/EOG). They are flags for later sensitivity analyses, not exclusions.

## Corrections relative to github.com/suvadeepmaiti/iSLEEPS (commit 02391ac)

| Reference code | Here |
|---|---|
| Hypnogram onset assumed = EDF start (`30·i`) → 0–29 s shift, 6–9 epochs in 5 files | Clock-aligned using xlsx start time; verified empirically (`check_alignment.py`) |
| Channel order from a Python set intersection → arbitrary, differs between M/A montages | Fixed canonical order with recorded source labels |
| Unused alias map would put F4/F3 into C4/C3 slots | Explicit per-channel alias lists |
| `mne.io.read_raw_edf` upsamples all channels to the highest rate, then resamples again | Each channel resampled once from its native rate |
| Variable channel count per file | Fixed 9 channels, NaN + mask when missing |
| SN28 duplicate kept | Excluded |
| ~95k one-epoch files, no epoch index/gaps | One file per patient with epoch index, segments, QC |
