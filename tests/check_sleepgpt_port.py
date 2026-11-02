"""Verify sleepgpt_c4.SleepGPTC4Encoder against the original SleepGPT MultiWayTransformer.

Loads models/SleepGPT/main/modules/multiway_transformer.py with lightweight stand-ins for its
logging/timm/GPU-monitor imports, builds `backbone_large_patch200` with actual_channels='C4', loads
the pretrained weights, and runs their pipeline (8-slot input, fixed normalization, get_fft, embed,
token-type embeddings, all-ones attention mask, 12 blocks, final norm) on real iSLEEPS C4 epochs.
"""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
REPO = repo_paths.SLEEPGPT_REPO  # needs einops (pip)


def load_reference():
    timm = types.ModuleType("timm"); timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.DropPath = lambda p: torch.nn.Identity()
    timm_layers.to_2tuple = lambda x: (x, x)
    timm_layers.trunc_normal_ = torch.nn.init.trunc_normal_
    timm_models.register_model = lambda f: f
    timm_models.layers = timm_layers
    pl = types.ModuleType("pytorch_lightning"); plu = types.ModuleType("pytorch_lightning.utilities")
    plr = types.ModuleType("pytorch_lightning.utilities.rank_zero"); plr.rank_zero_info = lambda *a, **k: None
    main = types.ModuleType("main"); main.__path__ = []
    modules = types.ModuleType("main.modules"); modules.__path__ = []
    transforms = types.ModuleType("main.transforms"); transforms.FFT_Transform = object
    adapter = types.ModuleType("main.modules.WearableAdapter")
    sys.modules.update({"timm": timm, "timm.models": timm_models, "timm.models.layers": timm_layers,
                        "pytorch_lightning": pl, "pytorch_lightning.utilities": plu,
                        "pytorch_lightning.utilities.rank_zero": plr, "pynvml": types.ModuleType("pynvml"),
                        "main": main, "main.modules": modules, "main.transforms": transforms,
                        "main.modules.WearableAdapter": adapter})
    spec = importlib.util.spec_from_file_location("main.modules.multiway_transformer",
                                                  REPO / "main" / "modules" / "multiway_transformer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    from sleepgpt_c4 import C4, CHECKPOINT, SleepGPTC4Encoder
    mwt = load_reference()
    config = {"drop_path_rate": 0.0, "random_choose_channels": 8, "actual_channels": "C4", "mask_ratio": 0.75,
              "epoch_duration": 30, "fs": 100, "fft_only": False, "time_only": False,
              "loss_names": {"mtm": 1, "itc": 0, "itm": 0}, "use_cb": False}
    ref = mwt.backbone_large_patch200(config=config).eval()
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    missing, unexpected = ref.load_state_dict({k[len("transformer."):]: v for k, v in sd.items()
                                               if k.startswith("transformer.")}, strict=True), None
    token_type = sd["token_type_embeddings.weight"]
    ours = SleepGPTC4Encoder(drop_path_rate=0.0).eval()

    with np.load(repo_paths.ISLEEPS_CACHE / "c4_100hz" / "SN1.npz") as d:
        x_uv = torch.from_numpy(d["x"][100:164, 0].astype(np.float32))
    mu = torch.tensor([-6.8460e-02, 1.9104e-01, 3.8937e-01, -2.0938e+00, 1.6496e-03, -4.8439e-05, 8.1125e-04, 7.1748e-05])
    std = torch.tensor([34.6887, 34.9556, 23.2826, 35.4035, 26.8738, 4.9272, 25.1366, 3.6142])
    with torch.no_grad():
        slots = torch.zeros(len(x_uv), 8, 3000)
        slots[:, C4] = x_uv
        epochs = (slots - mu[:, None]) / std[:, None]          # their normalize() for 8 channels
        epochs[:, [i for i in range(8) if i != C4]] = 0        # absent channels (dropped by actual_channels)
        fft, _ = ref.get_fft(epochs, torch.ones(len(x_uv), 8))
        res = ref.embed([epochs, fft], attn_mask=None, mask=False)
        x = res["x"]; n_time = res["x_len"]
        x = torch.cat([x[:, :n_time] + token_type[0], x[:, n_time:] + token_type[1]], dim=1)
        mask = torch.ones(x.shape[:2])
        for blk in ref.blocks:
            x = blk(x, mask=mask, modality_type="tf")
        ref_tokens = ref.norm(x)
        our_tokens = ours.tokens(x_uv)
    diff = (ref_tokens - our_tokens).abs().max().item()
    rel = diff / ref_tokens.abs().max().item()
    print(f"tokens {tuple(ref_tokens.shape)}; max |reference - port| = {diff:.3e} (relative {rel:.3e})")
    assert ref_tokens.shape == our_tokens.shape == (len(x_uv), 32, 768)
    assert rel < 1e-4, "Port does not match the original SleepGPT encoder"
    print("SleepGPT C4 port matches the original implementation.")


if __name__ == "__main__":
    main()
