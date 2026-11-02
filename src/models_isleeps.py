"""SE-ResNet18 + BiLSTM teacher for iSLEEPS; the student is the unchanged SleepStudentNet.

Port of SEresnet18_3LSTM from github.com/HimalayanSaswataBose/iSLEEPS_GeneralisationGapAndExplainability
(SHHS_to_iSLEEPS.ipynb / model_train_gradcam.ipynb, commit f9eb0cd), rewritten in plain PyTorch.
Kept: 1D SE-ResNet18 epoch encoder (kernel 15 stem, kernel 7 blocks, SE reduction 16, block
dropout 0.2, zero-initialised last BN), 512-d embedding, bidirectional LSTM (hidden 100,
3 layers), linear 200->2000->200->5 head without activations, 9-epoch context window.
Changed: returns logits (the original returned LogSoftmax, equivalent under cross-entropy),
the scored epoch is the window centre (the original used index ceil(9/2)=5), and the
unused fc/fc1/softmax layers of the encoder are dropped.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

from models import SleepStudentNet  # noqa: E402,F401  (re-exported, unchanged)


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.dropout = nn.Dropout(dropout)
        self.downsample = None if stride == 1 and inplanes == planes else nn.Sequential(
            nn.Conv1d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm1d(planes))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        out = self.se(self.bn2(self.conv2(out)))
        return torch.relu(out + identity)


class SEResNet18(nn.Module):
    """30 s epoch (channels, 3000) -> 512-d embedding."""
    def __init__(self, n_channels=1, layers=(2, 2, 2, 2), block_dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, 15, 2, 7, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1))
        blocks, inplanes = [], 64
        for planes, count, stride in zip((64, 128, 256, 512), layers, (1, 2, 2, 2)):
            for i in range(count):
                blocks.append(BasicBlock(inplanes, planes, stride if i == 0 else 1, block_dropout))
                inplanes = planes
        self.blocks = nn.Sequential(*blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(-1)


class SEResNetBiLSTM(nn.Module):
    """
    x: (B, window, channels, 3000), or (B, window, 3000) for one channel
    Returns: logits (B, 5) for the epoch at position `target`, and None (no carried state)
    """
    def __init__(self, n_classes=5, n_channels=1, window=9, target=4, lstm_hidden=100,
                 lstm_layers=3, dropout=0.1, block_dropout=0.2):
        super().__init__()
        if not 0 <= target < window:
            raise ValueError("target must index a position inside the window")
        self.n_channels, self.window, self.target = n_channels, window, target
        self.cnn = SEResNet18(n_channels, block_dropout=block_dropout)
        self.lstm = nn.LSTM(512, lstm_hidden, lstm_layers, dropout=dropout, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * lstm_hidden, 2000), nn.Linear(2000, 200),
                                  nn.Linear(200, n_classes))

    def forward(self, x, h=None):
        if x.ndim == 3:
            x = x.unsqueeze(2)
        B, W, C, L = x.shape
        if C != self.n_channels or W != self.window:
            raise ValueError(f"Expected (B, {self.window}, {self.n_channels}, L), got {tuple(x.shape)}")
        feats = self.cnn(x.reshape(B * W, C, L)).reshape(B, W, -1)
        out, _ = self.lstm(feats)
        return self.head(self.dropout(out[:, self.target])), None

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
