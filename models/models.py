import torch.nn as nn

class TinySeparator(nn.Module):
    def __init__(self, in_ch=8, hidden=64, out_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)

# ------------------- Simple Linear Separator (Baseline) ------------------- #
class LinearSeparator(nn.Module):
    def __init__(self, in_ch=8, out_ch=4):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)

# ------------------- Complicated ResidualBlock ------------------- #
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.act1 = nn.ReLU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.act2 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.act1(out)

        out = self.dropout(out)

        out = self.conv2(out)

        out = out + residual
        out = self.act2(out)

        return out


class HybridSeparator(nn.Module):
    """
    Linear spatial front-end + residual temporal refinement.

    Input:  (B, 8, T)
    Output: (B, 4, T)
    """
    def __init__(
        self,
        in_ch: int = 8,
        hidden: int = 64,
        out_ch: int = 4,
        num_blocks: int = 4,
        kernel_size: int = 9,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Linear spatial/front-end projection
        self.in_proj = nn.Conv1d(in_ch, hidden, kernel_size=1)

        # Temporal refinement
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden, kernel_size=kernel_size, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        # Output projection
        self.out_proj = nn.Conv1d(hidden, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x

class LSTMSeparator(nn.Module):
    """
    Input:  (B, 8, T)
    Output: (B, 4, T)

    Treats time as the sequence dimension and the 8 channels as features.
    """
    def __init__(
        self,
        in_ch: int = 8,
        hidden_size: int = 128,
        out_ch: int = 4,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.hidden_size = hidden_size
        self.out_ch = out_ch
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(in_ch, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_ch),
        )

    def forward(self, x):
        """
        x: (B, 8, T)
        returns: (B, 4, T)
        """
        # Convert from (B, C, T) to (B, T, C) for LSTM
        x = x.transpose(1, 2)

        # Project channel features before LSTM
        x = self.input_proj(x)

        # Sequence modeling
        x, _ = self.lstm(x)

        # Map each timestep to 4 output channels
        x = self.output_proj(x)

        # Convert back to (B, 4, T)
        x = x.transpose(1, 2)
        return x