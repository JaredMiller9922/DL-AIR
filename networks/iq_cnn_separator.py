import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils.conversion_helpers import channels_to_iq_view, iq_view_to_channels


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        groups = min(8, max(1, out_ch // 16))

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.gn2(self.conv2(x)))
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ConvBlock1D(in_ch, out_ch, kernel_size=5, stride=2, dropout=dropout)

    def forward(self, x):
        skip = x
        out = self.block(x)
        return out, skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        groups = min(8, max(1, out_ch // 16))
        self.upconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.act = nn.GELU()
        self.convblock = ConvBlock1D(out_ch * 2, out_ch, kernel_size=3, dropout=dropout)

    def forward(self, x, skip):
        x = self.act(self.gn(self.upconv(x)))
        if skip.shape[-1] != x.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                skip = skip[..., :x.shape[-1]]
            else:
                skip = F.pad(skip, (0, -diff))
        return self.convblock(torch.cat([x, skip], dim=1))


class MultichannelUNetSeparator(nn.Module):
    def __init__(self, base_channels=32, in_antennas=4, out_sources=2, dropout=0.08):
        super().__init__()
        self.in_ant = in_antennas
        self.out_src = out_sources

        input_groups = min(8, max(1, base_channels // 16))
        self.input_proj = nn.Conv1d(in_antennas * 2, base_channels, kernel_size=1, bias=False)
        self.input_gn = nn.GroupNorm(num_groups=input_groups, num_channels=base_channels)
        self.input_act = nn.GELU()
        self.input_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.down1 = DownsampleBlock(base_channels, base_channels, dropout=dropout)
        self.down2 = DownsampleBlock(base_channels, base_channels * 2, dropout=dropout)
        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4, dropout=dropout)

        self.bottle_conv1 = ConvBlock1D(base_channels * 4, base_channels * 4, kernel_size=3, dilation=1, dropout=dropout)
        self.bottle_conv2 = ConvBlock1D(base_channels * 4, base_channels * 4, kernel_size=3, dilation=2, dropout=dropout)
        self.bottle_conv3 = ConvBlock1D(base_channels * 4, base_channels * 4, kernel_size=3, dilation=4, dropout=dropout)

        self.up3 = UpsampleBlock(base_channels * 4, base_channels * 2, dropout=dropout)
        self.up2 = UpsampleBlock(base_channels * 2, base_channels, dropout=dropout)
        self.up1 = UpsampleBlock(base_channels, base_channels, dropout=dropout)

        self.final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.final_conv = nn.Conv1d(base_channels, out_sources * 2, kernel_size=1, bias=True)

    def forward(self, x):
        batch, antennas, time, iq = x.shape
        if antennas != self.in_ant or iq != 2:
            raise ValueError(f"Input must be (B, {self.in_ant}, T, 2), got {tuple(x.shape)}")

        x = x.permute(0, 1, 3, 2).contiguous().view(batch, antennas * 2, time)
        x = self.input_dropout(self.input_act(self.input_gn(self.input_proj(x))))

        d1, s1 = self.down1(x)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)

        bottleneck = self.bottle_conv1(d3)
        bottleneck = self.bottle_conv2(bottleneck)
        bottleneck = self.bottle_conv3(bottleneck)

        u3 = self.up3(bottleneck, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)

        out = self.final_conv(self.final_dropout(u1))
        return out.view(batch, self.out_src, 2, out.shape[-1]).permute(0, 1, 3, 2).contiguous()


class IQCNNSeparator(nn.Module):
    def __init__(self, in_ch=8, out_ch=4, base_channels=48, dropout=0.08):
        super().__init__()
        if in_ch % 2 != 0 or out_ch % 2 != 0:
            raise ValueError("IQCNNSeparator requires even in_ch and out_ch values")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.unet = MultichannelUNetSeparator(
            base_channels=base_channels,
            in_antennas=in_ch // 2,
            out_sources=out_ch // 2,
            dropout=dropout,
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_ch:
            raise ValueError(f"Expected input shape (B, {self.in_ch}, T), got {tuple(x.shape)}")
        return iq_view_to_channels(self.unet(channels_to_iq_view(x)))
