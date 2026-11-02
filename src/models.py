"""Repository architectures with configurable input channels. See provenance.json."""
import torch
import torch.nn as nn
EPOCH_SAMP = 3000


class TinySleepNet(nn.Module):
    """
    PyTorch re-implementation of TinySleepNet.
    """
    def __init__(self, n_classes=5, fs=100, seq_len=20, dropout=0.5, n_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        lf = fs // 2   # 50 samples
        self.cnn_large = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=lf, stride=lf//2, padding=lf//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.Conv1d(128, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        # Small filter branch (captures fast spindles/K-complexes)
        sf = fs // 4   # 25 samples
        self.cnn_small = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=sf, stride=sf//2, padding=sf//2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.Conv1d(128, 128, kernel_size=6, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_drop = nn.Dropout(p=dropout)

        # Compute CNN output dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, EPOCH_SAMP)
            l_out = self.cnn_large(dummy).flatten(1)
            s_out = self.cnn_small(dummy).flatten(1)
            cnn_dim = l_out.shape[1] + s_out.shape[1]

        # LSTM backend 
        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm_drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward_cnn(self, x):
        """x: (B, channels, 3000) → (B, cnn_dim)"""
        l = self.cnn_large(x).flatten(1)
        s = self.cnn_small(x).flatten(1)
        return self.cnn_drop(torch.cat([l, s], dim=1))

    def forward(self, x, h=None):
        """
        x: (B, seq_len, channels, 3000), or (B, seq_len, 3000) for EEG only
        Returns: logits (B, seq_len, 5), hidden state h
        """
        if x.ndim == 3:
            x = x.unsqueeze(2)
        B, T, C, L = x.shape
        if C != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {C}")
        # CNN over all epochs in parallel
        x_flat = x.reshape(B * T, C, L)
        feats = self.forward_cnn(x_flat)         # (B*T, cnn_dim)
        feats = feats.reshape(B, T, -1)          # (B, T, cnn_dim)
        # LSTM over sequence
        lstm_out, h_new = self.lstm(feats, h)    # (B, T, 128)
        lstm_out = self.lstm_drop(lstm_out)
        logits = self.fc(lstm_out)               # (B, T, 5)
        return logits, h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SleepStudentNet(nn.Module):
    """
    Lightweight CNN-GRU student for edge deployment.
    Input  : single epoch (channels, 3000)
    Output : 5-class logits
    """
    def __init__(self, n_classes=5, gru_hidden=64, dropout=0.3, n_channels=1):
        super().__init__()
        self.n_channels = n_channels

        self.cnn = nn.Sequential(
            # Layer 1: large receptive field for slow waves
            nn.Conv1d(n_channels, 16, kernel_size=50, stride=5, padding=25),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),

            # Layer 2: mid-range features
            nn.Conv1d(16, 32, kernel_size=10, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),

            # Layer 3: fine features
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),   # fixed output length
        )

        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(32, gru_hidden, batch_first=True, num_layers=1)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(self, x, h=None):
        """
        x : (B, channels, 3000), or (B, 3000) for EEG only
        h : GRU hidden state — pass across consecutive epochs for temporal context
        Returns: logits (B, n_classes), h_new
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.shape[1]}")
        feat = self.cnn(x)                    # (B, 32, 32)
        feat = feat.permute(0, 2, 1)          # (B, 32, 32) — seq for GRU
        feat = self.dropout(feat)
        out, h_new = self.gru(feat, h)        # (B, 32, 64)
        logits = self.fc(out[:, -1, :])       # last timestep → (B, 5)
        return logits, h_new

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
