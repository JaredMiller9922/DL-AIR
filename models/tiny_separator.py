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