"""Multichannel context teachers for the privileged-channel experiment (students stay C4-only).

SleepGPTMultiEncoder generalizes sleepgpt_c4.SleepGPTC4Encoder from the C4 slot to any subset of
SleepGPT's pretraining slots, exactly as their `actual_channels` mode does (here 'shhs' = C3, C4,
EMG, EOG): grouped patch/FFT convolutions use each slot's own pretrained weights, tokens are
channel-major, absolute position embeddings repeat per channel and channel embeddings repeat per
patch, the time/FFT MLP split sits at 1 + 15 * channels, and each slot uses its fixed
normalization from their main/transforms/transform.py. Verified by check_sleepgpt_multi.py.

Both teachers expose embed(x) -> per-epoch embedding and forward_embeddings(e) -> logits, so the
engine can embed each epoch once per batch and share it across overlapping context windows.
"""
from pathlib import Path
import sys

import torch
import torch.nn as nn

from sleepgpt_c4 import CHECKPOINT, DEPTH, DIM, HEADS, PATCH, PATCHES, TF_START, Block, CHANNELS

from models_isleeps import SEResNet18  # noqa: E402

SLOT_MEAN = (-6.8460e-02, 1.9104e-01, 3.8937e-01, -2.0938e+00, 1.6496e-03, -4.8439e-05, 8.1125e-04, 7.1748e-05)
SLOT_STD = (34.6887, 34.9556, 23.2826, 35.4035, 26.8738, 4.9272, 25.1366, 3.6142)


class MultiBlock(Block):
    def __init__(self, dim, heads, shared_mlp, drop_path, time_len):
        super().__init__(dim, heads, shared_mlp, drop_path)
        self.time_len = time_len

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        if self.shared:
            return x + self.drop_path(self.gamma_2 * self.mlp_tf(self.norm2_tf(x)))
        t, f = x[:, :1 + self.time_len], x[:, 1 + self.time_len:]
        t = t + self.drop_path(self.gamma_2 * self.mlp_time(self.norm2_time(t)))
        f = f + self.drop_path(self.gamma_2 * self.mlp_fft(self.norm2_fft(f)))
        return torch.cat([t, f], dim=1)


class SleepGPTMultiEncoder(nn.Module):
    """x: (M, C, 3000) at 100 Hz in uV, channels in `slots` order -> (M, 1536)."""
    def __init__(self, slots=("C3", "C4", "EMG", "EOG"), drop_path_rate=0.2, pretrained=True):
        super().__init__()
        self.slots = [CHANNELS.index(s) for s in slots]
        C = len(self.slots)
        self.n_channels = C
        self.proj = nn.Conv1d(C, C * DIM, PATCH, PATCH, groups=C)
        self.fft_proj = nn.Conv2d(C, C * DIM, (2, 100), (2, 100), groups=C)
        self.cls_token, self.fft_cls_token = nn.Parameter(torch.zeros(1, 1, DIM)), nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, PATCHES, DIM))
        self.cls_token_pos_embed = nn.Parameter(torch.zeros(1, 1, DIM))
        self.channel_embed = nn.Parameter(torch.zeros(1, C, DIM))
        self.token_type_embeddings = nn.Embedding(2, DIM)
        rates = torch.linspace(0, drop_path_rate, DEPTH).tolist()
        self.blocks = nn.ModuleList(MultiBlock(DIM, HEADS, i >= TF_START, rates[i], PATCHES * C) for i in range(DEPTH))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)
        self.register_buffer("mean", torch.tensor([SLOT_MEAN[s] for s in self.slots]).view(1, C, 1), persistent=False)
        self.register_buffer("std", torch.tensor([SLOT_STD[s] for s in self.slots]).view(1, C, 1), persistent=False)
        self.register_buffer("window", torch.hann_window(PATCH), persistent=False)
        if pretrained:
            self.load_pretrained()

    def load_pretrained(self, path=CHECKPOINT):
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        rows = torch.cat([torch.arange(s * DIM, (s + 1) * DIM) for s in self.slots])
        state = {
            "proj.weight": sd["transformer.patch_embed.proj.weight"][rows],
            "proj.bias": sd["transformer.patch_embed.proj.bias"][rows],
            "fft_proj.weight": sd["transformer.patch_embed.fft_proj.weight"][rows],
            "fft_proj.bias": sd["transformer.patch_embed.fft_proj.bias"][rows],
            "channel_embed": sd["transformer.channel_embed"][:, self.slots],
            "token_type_embeddings.weight": sd["token_type_embeddings.weight"],
            **{k: sd[f"transformer.{k}"] for k in ("cls_token", "fft_cls_token", "pos_embed",
                                                    "cls_token_pos_embed", "norm.weight", "norm.bias")},
            **{k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.blocks.")},
        }
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and not missing, (missing, unexpected)

    def spectrogram(self, x):
        M, C, L = x.shape
        spec = torch.stft(x.reshape(M * C, L).float(), 256, 100, PATCH, self.window, return_complex=True).abs()
        spec = (10 * torch.log10(spec[:, :100, 1:] + 1e-8)).transpose(-2, -1)
        mean, std = spec.mean(-1, keepdim=True), spec.std(-1, keepdim=True)
        spec = torch.where(std != 0, (spec - mean) / std, torch.zeros_like(spec))
        return spec.reshape(M, C, 30, 100)

    def tokens(self, x):
        M, C, _ = x.shape
        x = (x.float() - self.mean) / self.std
        t = self.proj(x).reshape(M, C, DIM, PATCHES).permute(0, 1, 3, 2).reshape(M, C * PATCHES, DIM)
        f = self.fft_proj(self.spectrogram(x).to(t.dtype)).squeeze(-1)
        f = f.reshape(M, C, DIM, PATCHES).permute(0, 1, 3, 2).reshape(M, C * PATCHES, DIM)
        pos = self.pos_embed.repeat(1, C, 1) + self.channel_embed.repeat_interleave(PATCHES, dim=1)
        cls = self.cls_token_pos_embed
        t = torch.cat([self.cls_token.expand(M, -1, -1) + cls, t + pos], 1) + self.token_type_embeddings.weight[0]
        f = torch.cat([self.fft_cls_token.expand(M, -1, -1) + cls, f + pos], 1) + self.token_type_embeddings.weight[1]
        h = torch.cat([t, f], dim=1)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, x):
        h = self.tokens(x)
        n = PATCHES * self.n_channels
        return torch.cat([h[:, 1:1 + n].mean(1), h[:, 2 + n:].mean(1)], dim=-1)


class ContextHead(nn.Module):
    """BiLSTM over the window + linear 200->2000->200->5 on the centre, as in the earlier teachers."""
    def __init__(self, in_dim, target, lstm_hidden=100, lstm_layers=3, dropout=0.1, n_classes=5):
        super().__init__()
        self.target = target
        self.lstm = nn.LSTM(in_dim, lstm_hidden, lstm_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * lstm_hidden, 2000), nn.Linear(2000, 200), nn.Linear(200, n_classes))

    def forward(self, e):
        out, _ = self.lstm(e.float())
        return self.head(self.dropout(out[:, self.target]))


class SleepGPTMultiTeacher(nn.Module):
    def __init__(self, window=21, target=10, slots=("C3", "C4", "EMG", "EOG"), drop_path_rate=0.2, pretrained=True, **head):
        super().__init__()
        self.window, self.target = window, target
        self.encoder = SleepGPTMultiEncoder(slots, drop_path_rate, pretrained)
        self.context = ContextHead(2 * DIM, target, **head)

    def embed(self, x):
        return self.encoder(x)

    def forward_embeddings(self, e):
        return self.context(e)

    def parameter_groups(self, base_lr, layer_decay=0.75, weight_decay=0.05):
        """SleepGPT fine-tuning: layer-wise lr decay; new context head at the base lr."""
        groups = {}
        for name, p in self.named_parameters():
            if name.startswith("encoder."):
                n = name[len("encoder."):]
                scale = (layer_decay ** (DEPTH - int(n.split(".")[1])) if n.startswith("blocks.") else
                         1.0 if n.startswith("norm.") else layer_decay ** (DEPTH + 1))
                no_decay = p.ndim == 1 or n in ("cls_token", "fft_cls_token", "pos_embed", "cls_token_pos_embed",
                                                "channel_embed")
            else:
                scale, no_decay = 1.0, p.ndim == 1
            groups.setdefault((base_lr * scale, 0.0 if no_decay else weight_decay), []).append(p)
        return [{"params": ps, "lr": lr, "initial_lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SEResNetMultiTeacher(nn.Module):
    """SE-ResNet18 epoch encoder (from scratch) on the same 4 channels, normalized with training-fold stats."""
    def __init__(self, mean_uV, std_uV, window=21, target=10, block_dropout=0.2, **head):
        super().__init__()
        self.window, self.target = window, target
        C = len(mean_uV)
        self.register_buffer("mean", torch.tensor(mean_uV, dtype=torch.float32).view(1, C, 1))
        self.register_buffer("std", torch.tensor(std_uV, dtype=torch.float32).view(1, C, 1))
        self.encoder = SEResNet18(C, block_dropout=block_dropout)
        self.context = ContextHead(512, target, **head)

    def embed(self, x):
        return self.encoder((x.float() - self.mean) / self.std)

    def forward_embeddings(self, e):
        return self.context(e)

    def parameter_groups(self, base_lr, weight_decay=1e-4):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr,
                 "initial_lr": base_lr, "weight_decay": weight_decay}]

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
