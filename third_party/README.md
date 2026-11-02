# SleepGPT (not included)

The teacher's encoder is SleepGPT (Huang et al., 2026), a multiway transformer pretrained on large PSG collections.
`src/sleepgpt_c4.py` and `src/teachers_multi.py` re-implement its encoder in plain PyTorch and load its released
checkpoint; neither the SleepGPT code nor its weights are redistributed here.

1. Obtain the SleepGPT repository and its pretrained checkpoint from the authors.
2. Place it at `third_party/SleepGPT/` (or set `SLEEPGPT_DIR`), so that the checkpoint is at
   `third_party/SleepGPT/checkpoint/ModelCheckpoint.ckpt`.
3. Optional: `python tests/check_sleepgpt_port.py` compares the re-implementation with the original module on real
   epochs (needs `einops` and the original repository).
