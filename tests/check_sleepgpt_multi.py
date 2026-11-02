"""Verify teachers_multi.SleepGPTMultiEncoder against the original SleepGPT module ('shhs' channels)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
from check_sleepgpt_port import load_reference  # noqa: E402
from sleepgpt_c4 import CHECKPOINT
from teachers_multi import SLOT_MEAN, SLOT_STD, SleepGPTMultiEncoder

HERE = Path(__file__).resolve().parent


def main():
    mwt = load_reference()
    config = {"drop_path_rate": 0.0, "random_choose_channels": 8, "actual_channels": "shhs", "mask_ratio": 0.75,
              "epoch_duration": 30, "fs": 100, "fft_only": False, "time_only": False,
              "loss_names": {"mtm": 1, "itc": 0, "itm": 0}, "use_cb": False}
    ref = mwt.backbone_large_patch200(config=config).eval()
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    ref.load_state_dict({k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}, strict=True)
    token_type = sd["token_type_embeddings.weight"]
    ours = SleepGPTMultiEncoder(drop_path_rate=0.0).eval()
    src = repo_paths.PREPROCESSED["isleeps"] / "SN57.npz"
    with np.load(src) as d:
        names = d["channels"].tolist()
        x4 = torch.from_numpy(d["x"][200:248][:, [names.index(c) for c in ("C3-M2", "C4-M1", "EMG", "E1-M2")]])
    mu, std = torch.tensor(SLOT_MEAN), torch.tensor(SLOT_STD)
    with torch.no_grad():
        slots = torch.zeros(len(x4), 8, 3000)
        slots[:, :4] = x4                                   # slots 0-3 = C3, C4, EMG, EOG
        epochs = (slots - mu[:, None]) / std[:, None]
        epochs[:, 4:] = 0
        fft, _ = ref.get_fft(epochs, torch.ones(len(x4), 8))
        res = ref.embed([epochs, fft], attn_mask=None, mask=False)
        x, n_time = res["x"], res["x_len"]
        x = torch.cat([x[:, :n_time] + token_type[0], x[:, n_time:] + token_type[1]], dim=1)
        for blk in ref.blocks:
            x = blk(x, mask=torch.ones(x.shape[:2]), modality_type="tf")
        ref_tokens, our_tokens = ref.norm(x), ours.tokens(x4)
    rel = (ref_tokens - our_tokens).abs().max().item() / ref_tokens.abs().max().item()
    print(f"tokens {tuple(ref_tokens.shape)}; max relative |reference - port| = {rel:.3e}")
    assert ref_tokens.shape == our_tokens.shape == (len(x4), 122, 768) and rel < 1e-4
    print("SleepGPT C3/C4/EMG/EOG port matches the original implementation.")


if __name__ == "__main__":
    main()
