"""Single-channel (C4) port of the pretrained SleepGPT encoder + BiLSTM context teacher.

Source: github.com/LordXX505/SleepGPT (commit fcbb2c7; no licence file in the repository; paper
Nature Communications 2026, doi:10.1038/s41467-025-67970-4), `backbone_large_patch200` from
main/modules/multiway_transformer.py with the pretrained checkpoint
models/SleepGPT/checkpoint/ModelCheckpoint.ckpt (epoch 79, masked time/frequency reconstruction on
SHHS-1/2, PhysioNet-2018, SleepEEGfMRI and Young). Ported: the C4 path of their
`actual_channels='C4'` mode, which drops all other channel tokens (equivalent to attention-masking
them), keeps 1 time CLS + 15 time patches + 1 FFT CLS + 15 FFT patches, and uses fixed C4
normalization (mean 0.19104 uV, std 34.9556 uV, from their main/transforms/transform.py).
Verified against the original module by check_sleepgpt_port.py.
"""
import repo_paths
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CHECKPOINT = repo_paths.SLEEPGPT_CHECKPOINT  # not distributed here, see third_party/README.md
CHANNELS = ("C3", "C4", "EMG", "EOG", "F3", "Fpz", "O1", "Pz")  # pretraining channel slots
C4 = CHANNELS.index("C4")
C4_MEAN_UV, C4_STD_UV = 0.19104, 34.9556
DIM, DEPTH, HEADS, TF_START, PATCHES, PATCH = 768, 12, 16, 10, 15, 200


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = F.linear(x, self.qkv.weight, bias).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """Multiway block: shared attention; separate time/FFT MLPs below TF_START, shared above."""
    def __init__(self, dim, heads, shared_mlp, drop_path):
        super().__init__()
        self.shared = shared_mlp
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.gamma_1, self.gamma_2 = nn.Parameter(0.1 * torch.ones(dim)), nn.Parameter(0.1 * torch.ones(dim))
        if shared_mlp:
            self.norm2_tf, self.mlp_tf = nn.LayerNorm(dim, eps=1e-6), Mlp(dim)
        else:
            self.norm2_time, self.mlp_time = nn.LayerNorm(dim, eps=1e-6), Mlp(dim)
            self.norm2_fft, self.mlp_fft = nn.LayerNorm(dim, eps=1e-6), Mlp(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        if self.shared:
            return x + self.drop_path(self.gamma_2 * self.mlp_tf(self.norm2_tf(x)))
        t, f = x[:, :1 + PATCHES], x[:, 1 + PATCHES:]
        t = t + self.drop_path(self.gamma_2 * self.mlp_time(self.norm2_time(t)))
        f = f + self.drop_path(self.gamma_2 * self.mlp_fft(self.norm2_fft(f)))
        return torch.cat([t, f], dim=1)


class SleepGPTC4Encoder(nn.Module):
    """x: (M, 3000) C4 epochs at 100 Hz in uV -> (M, 1536) = [mean time tokens, mean FFT tokens]."""
    def __init__(self, drop_path_rate=0.2, pretrained=True):
        super().__init__()
        self.proj = nn.Conv1d(1, DIM, PATCH, PATCH)
        self.fft_proj = nn.Conv2d(1, DIM, (2, 100), (2, 100))
        self.cls_token, self.fft_cls_token = nn.Parameter(torch.zeros(1, 1, DIM)), nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, PATCHES, DIM))
        self.cls_token_pos_embed = nn.Parameter(torch.zeros(1, 1, DIM))
        self.channel_embed = nn.Parameter(torch.zeros(1, 1, DIM))
        self.token_type_embeddings = nn.Embedding(2, DIM)
        rates = torch.linspace(0, drop_path_rate, DEPTH).tolist()
        self.blocks = nn.ModuleList(Block(DIM, HEADS, i >= TF_START, rates[i]) for i in range(DEPTH))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)
        self.register_buffer("window", torch.hann_window(PATCH), persistent=False)
        if pretrained:
            self.load_pretrained()

    def load_pretrained(self, path=CHECKPOINT):
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        g = slice(C4 * DIM, (C4 + 1) * DIM)  # grouped conv: output channels of the C4 group
        state = {
            "proj.weight": sd["transformer.patch_embed.proj.weight"][g],
            "proj.bias": sd["transformer.patch_embed.proj.bias"][g],
            "fft_proj.weight": sd["transformer.patch_embed.fft_proj.weight"][g],
            "fft_proj.bias": sd["transformer.patch_embed.fft_proj.bias"][g],
            "channel_embed": sd["transformer.channel_embed"][:, C4:C4 + 1],
            "token_type_embeddings.weight": sd["token_type_embeddings.weight"],
            **{k: sd[f"transformer.{k}"] for k in ("cls_token", "fft_cls_token", "pos_embed",
                                                    "cls_token_pos_embed", "norm.weight", "norm.bias")},
            **{k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.blocks.")},
        }
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and not missing, (missing, unexpected)

    def spectrogram(self, x):
        """Their get_fft: STFT (n_fft 256, hop 100, Hann 200), first 100 bins, frames 1..30, dB, per-frame z-score."""
        spec = torch.stft(x.float(), 256, 100, PATCH, self.window, return_complex=True).abs()[:, :100, 1:]
        spec = (10 * torch.log10(spec + 1e-8)).transpose(-2, -1)
        mean, std = spec.mean(-1, keepdim=True), spec.std(-1, keepdim=True)
        return torch.where(std != 0, (spec - mean) / std, torch.zeros_like(spec))

    def tokens(self, x):
        x = (x.float() - C4_MEAN_UV) / C4_STD_UV
        t = self.proj(x.unsqueeze(1)).transpose(1, 2)                                   # (M, 15, D)
        f = self.fft_proj(self.spectrogram(x).unsqueeze(1).to(t.dtype)).squeeze(-1).transpose(1, 2)
        M = x.shape[0]
        t = torch.cat([self.cls_token.expand(M, -1, -1) + self.cls_token_pos_embed,
                       t + self.pos_embed + self.channel_embed], dim=1) + self.token_type_embeddings.weight[0]
        f = torch.cat([self.fft_cls_token.expand(M, -1, -1) + self.cls_token_pos_embed,
                       f + self.pos_embed + self.channel_embed], dim=1) + self.token_type_embeddings.weight[1]
        x = torch.cat([t, f], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)                                                              # (M, 32, D)

    def forward(self, x):
        h = self.tokens(x)
        return torch.cat([h[:, 1:1 + PATCHES].mean(1), h[:, 2 + PATCHES:].mean(1)], dim=-1)


class SleepGPTContextTeacher(nn.Module):
    """
    x: (B, window, 3000) C4 epochs at 100 Hz in uV
    Returns: logits (B, 5) for the epoch at position `target`, None
    Context head as in the SE-ResNet and BIOT-BiLSTM teachers (BiLSTM hidden 100, 3 layers).
    """
    def __init__(self, n_classes=5, window=9, target=4, lstm_hidden=100, lstm_layers=3, dropout=0.1,
                 drop_path_rate=0.2, pretrained=True):
        super().__init__()
        self.window, self.target = window, target
        self.encoder = SleepGPTC4Encoder(drop_path_rate, pretrained)
        self.lstm = nn.LSTM(2 * DIM, lstm_hidden, lstm_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * lstm_hidden, 2000), nn.Linear(2000, 200), nn.Linear(200, n_classes))

    def embed(self, x):
        return self.encoder(x)

    def forward_embeddings(self, e):
        out, _ = self.lstm(e.float())
        return self.head(self.dropout(out[:, self.target]))

    def forward(self, x, h=None):
        B, W, L = x.shape
        return self.forward_embeddings(self.embed(x.reshape(B * W, L)).reshape(B, W, -1)), None

    def parameter_groups(self, base_lr, head_lr, layer_decay=0.75, weight_decay=0.05):
        """Layer-wise lr decay as in SleepGPT fine-tuning: embeddings get decay^(depth+1), block i decay^(depth-i)."""
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder."):
                n = name[len("encoder."):]
                if n.startswith("blocks."):
                    scale = layer_decay ** (DEPTH - int(n.split(".")[1]))
                elif n.startswith("norm."):
                    scale = 1.0
                else:
                    scale = layer_decay ** (DEPTH + 1)
                lr = base_lr * scale
                no_decay = p.ndim == 1 or n in ("cls_token", "fft_cls_token", "pos_embed", "cls_token_pos_embed",
                                                "channel_embed")
            else:
                lr, no_decay = head_lr, p.ndim == 1
            key = (lr, 0.0 if no_decay else weight_decay)
            groups.setdefault(key, []).append(p)
        return [{"params": ps, "lr": lr, "initial_lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
