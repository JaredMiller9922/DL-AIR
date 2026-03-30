import torch.nn as nn

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
        self.dropout_rate = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(in_ch, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

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
            nn.Dropout(dropout),
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
