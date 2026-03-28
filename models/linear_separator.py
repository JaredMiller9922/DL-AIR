import torch.nn as nn

class LinearSeparator(nn.Module):
    def __init__(self, in_ch=8, out_ch=4):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)

