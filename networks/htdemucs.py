import torch
import torch.nn as nn
import torch.nn.functional as F

class RFConvBlock(nn.Module):
    """
    Standard Demucs Convolutional Block: Conv1d -> GLU -> LayerNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=8, stride=4, padding=2, is_encoder=True):
        super().__init__()
        # GLU doubles the channel dimension output of the convolution
        conv_out_channels = out_channels * 2 if is_encoder else out_channels * 2
        
        if is_encoder:
            self.conv = nn.Conv1d(in_channels, conv_out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, conv_out_channels, kernel_size, stride, padding)
            
        self.norm = nn.GroupNorm(1, out_channels) # GroupNorm(1) is equivalent to LayerNorm for channels
        
    def forward(self, x):
        x = self.conv(x)
        # Apply GLU over the channel dimension
        x = F.glu(x, dim=1)
        x = self.norm(x)
        x = F.relu(x)
        return x

class CrossDomainAttention(nn.Module):
    """
    Cross-references the spatial/spectral domain with the temporal domain.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.GroupNorm(1, channels)
        
    def forward(self, query, key_value):
        # PyTorch MHA expects (Batch, Seq, Channels) if batch_first=True
        # Inputs are (Batch, Channels, Seq), so we permute
        q = query.permute(0, 2, 1)
        kv = key_value.permute(0, 2, 1)
        
        attn_out, _ = self.mha(q, kv, kv)
        attn_out = attn_out.permute(0, 2, 1) # Back to (Batch, Channels, Seq)
        
        # Residual connection + norm
        return self.norm(query + attn_out)

class RFHTDemucs(nn.Module):
    def __init__(
        self, 
        in_ch=2,          # Number of input channels (e.g., 2 for mixture I/Q)
        out_ch=4,         # Number of output channels (e.g., 4 for srcA_I, srcA_Q, srcB_I, srcB_Q)
        base_channels=64, # Starting channel depth
        depth=4,          # Number of U-Net layers
        kernel_size=8,    # Convolution kernel
        stride=4,         # Downsampling factor
        bottleneck_layers=4 # Depth of the inner transformer
    ):
        super().__init__()
        self.depth = depth
        
        # --- ENCODERS ---
        self.time_encoders = nn.ModuleList()
        self.freq_encoders = nn.ModuleList()
        
        # --- CROSS-DOMAIN INTERACTION ---
        self.cross_t2f = nn.ModuleList()
        self.cross_f2t = nn.ModuleList()
        
        # --- DECODERS ---
        self.time_decoders = nn.ModuleList()
        self.freq_decoders = nn.ModuleList()
        
        ch_in_time = in_ch
        ch_in_freq = in_ch * 2 # Usually freq has real/imag or mag/phase, so 2x channels
        ch_out = base_channels
        
        # Build U-Net downsampling paths
        for i in range(depth):
            padding = (kernel_size - stride) // 2
            
            self.time_encoders.append(RFConvBlock(ch_in_time, ch_out, kernel_size, stride, padding, True))
            self.freq_encoders.append(RFConvBlock(ch_in_freq, ch_out, kernel_size, stride, padding, True))
            
            # Cross attention: Time attending to Freq, Freq attending to Time
            self.cross_t2f.append(CrossDomainAttention(ch_out))
            self.cross_f2t.append(CrossDomainAttention(ch_out))
            
            ch_in_time = ch_out
            ch_in_freq = ch_out
            ch_out *= 2
            
        # --- BOTTLENECK TRANSFORMER ---
        bottleneck_dim = ch_in_time # Max channel depth
        encoder_layer = nn.TransformerEncoderLayer(d_model=bottleneck_dim, nhead=8, batch_first=True)
        self.bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=bottleneck_layers)
        
        # Build U-Net upsampling paths
        ch_in = bottleneck_dim
        for i in range(depth):
            ch_out = ch_in // 2 if i < depth - 1 else out_ch // 2 # out_ch//2 per branch
            padding = (kernel_size - stride) // 2
            
            # Note: input channels double because of the U-Net skip connections
            self.time_decoders.append(RFConvBlock(ch_in * 2, ch_out, kernel_size, stride, padding, False))
            self.freq_decoders.append(RFConvBlock(ch_in * 2, ch_out, kernel_size, stride, padding, False))
            
            ch_in = ch_out
            
        # Final projection to exact output channels
        self.final_proj = nn.Conv1d(out_ch, out_ch, kernel_size=1)

    def forward(self, x_time, x_freq):
        """
        x_time: (B, in_ch, T)
        x_freq: (B, in_ch*2, F) -> Output of an STFT
        """
        time_skips = []
        freq_skips = []
        
        # --- ENCODER FORWARD ---
        for i in range(self.depth):
            x_time = self.time_encoders[i](x_time)
            x_freq = self.freq_encoders[i](x_freq)
            
            # Apply Cross-Domain Attention
            x_time_cross = self.cross_t2f[i](x_time, x_freq)
            x_freq_cross = self.cross_f2t[i](x_freq, x_time)
            
            x_time, x_freq = x_time_cross, x_freq_cross
            
            time_skips.append(x_time)
            freq_skips.append(x_freq)
            
        # --- BOTTLENECK ---
        # Only applying to time domain for this template, 
        # HTDemucs combines or bottlenecks both depending on the exact variant.
        B, C, T = x_time.shape
        x_time = x_time.permute(0, 2, 1) # (B, T, C)
        x_time = self.bottleneck(x_time)
        x_time = x_time.permute(0, 2, 1) # (B, C, T)
        
        # --- DECODER FORWARD ---
        for i in range(self.depth):
            # Pop the matching skip connection
            skip_time = time_skips.pop()
            skip_freq = freq_skips.pop()
            
            # Concatenate skip connection (Channel dim)
            x_time = torch.cat([x_time, skip_time], dim=1)
            x_freq = torch.cat([x_freq, skip_freq], dim=1)
            
            x_time = self.time_decoders[i](x_time)
            x_freq = self.freq_decoders[i](x_freq)
            
        # Recombine branches (Assume an ISTFT will eventually process x_freq)
        # For now, we concatenate the branches to output the final separated sources
        out = torch.cat([x_time, x_freq], dim=1)
        return self.final_proj(out)