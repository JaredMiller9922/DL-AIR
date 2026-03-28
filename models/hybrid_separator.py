import torch.nn as nn

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