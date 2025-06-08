import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def center_crop_or_pad(x, target_h, target_w):
    B, C, H, W = x.shape

    # --- 세로축(H) ---
    if H > target_h:
        # crop
        diff = H - target_h
        start = diff // 2
        end = start + target_h
        x = x[:, :, start:end, :]
    elif H < target_h:
        # pad
        diff = target_h - H
        pad_before = diff // 2
        pad_after = diff - pad_before
        # (left, right, top, bottom) = (0,0, pad_before, pad_after) for H dimension
        x = F.pad(x, (0, 0, pad_before, pad_after))

    # --- 가로축(W) ---
    if W > target_w:
        # crop
        diff = W - target_w
        start = diff // 2
        end = start + target_w
        x = x[:, :, :, start:end]
    elif W < target_w:
        # pad
        diff = target_w - W
        pad_before = diff // 2
        pad_after = diff - pad_before
        # (left, right, top, bottom) = (pad_before, pad_after, 0, 0) for W dimension
        x = F.pad(x, (pad_before, pad_after, 0, 0))

    return x


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = DoubleConv2D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)  # (H,W) -> ceil(H/2), ceil(W/2)
        x = self.conv(x)
        return x


class Up2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, 
            in_ch // 2, 
            kernel_size=2, 
            stride=2,
            output_padding=0  # crop/pad로 보정할 것이므로 0
        )
        self.conv = DoubleConv2D(in_ch // 2 + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # (Hup, Wup) = (2*Hd, 2*Wd)
        # center_crop_or_pad로 skip 크기에 맞춤
        # skip.shape[2] = Hs, skip.shape[3] = Ws
        x = center_crop_or_pad(x, skip.shape[2], skip.shape[3])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, H, W)
        y = x.mean(dim=(2,3), keepdim=True)     # squeeze
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y                             # excite

class DeepSleepAttn2D(nn.Module):
    def __init__(self, in_channels=9, base_ch=16, num_layers=4, 
                 transformer_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        # Encoder
        self.inc = DoubleConv2D(in_channels, base_ch)
        self.downs = nn.ModuleList()
        ch = base_ch
        for i in range(num_layers):
            # self.downs.append(Down2D(ch, ch*2))
            self.downs.append(nn.Sequential(
                Down2D(ch, ch*2),
                nn.Dropout2d(p=dropout)
            ))
            ch *= 2
        
        # Bottleneck dilated conv + SE
        self.bot = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            SEBlock(ch),
            nn.Dropout2d(p=dropout)
        )
        
        # Transformer Encoder
        # We'll flatten H dimension, treat time axis as sequence:
        # assume after downs: feature map is (B, ch, H', W')
        encoder_layer = TransformerEncoderLayer(d_model=ch, nhead=nhead, dim_feedforward=ch*2, dropout=0.1)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Decoder with SE and deep supervision heads
        self.ups = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(num_layers-1, -1, -1):
            prev_ch = ch
            skip_ch = base_ch * (2**i)
            self.ups.append(Up2D(prev_ch, skip_ch))
            # Deep supervision head
            # self.heads.append(nn.Conv2d(skip_ch, 1, kernel_size=1))
            self.heads.append(nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(skip_ch, 1, kernel_size=1)
            ))
            ch = skip_ch
        
        self.freq_conv = nn.Conv1d(26, 1, kernel_size=1)
        self.final_act = nn.Sigmoid()
    
    def forward(self, x, train=False, act=True, freq=False):
        # Encoder
        _, _, FREQ_BINS, TIME_BINS = x.shape
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bot(x)  # (B, C, H', W')
        
        B, C, H, W = x.shape
        # Prepare for transformer: flatten H dim, sequence along W
        x_seq = x.permute(0,2,3,1).reshape(B*H, W, C)  # (B*H, W, C)
        # Transformer expects (seq_len, batch, features)
        x_seq = x_seq.permute(1,0,2)                   # (W, B*H, C)
        x_seq = self.transformer(x_seq)                # (W, B*H, C)
        x_seq = x_seq.permute(1,0,2).reshape(B, H, W, C).permute(0,3,1,2)  # back -> (B, C, H, W)
        
        # Decoder with deep supervision
        outputs = []
        for up, head, skip in zip(self.ups, self.heads, reversed(skips[:-1])):
            x = up(x, skip)
            out = head(x)
            if out.shape[2] != FREQ_BINS or out.shape[3] != TIME_BINS:
                out = F.interpolate(out, size=(FREQ_BINS, TIME_BINS), mode='bilinear', align_corners=False)
            # out = self.final_act(out)
            outputs.append(out)

        if freq:
            outputs[-1] = self.freq_conv(outputs[-1].squeeze(1))

        if act:
            final = self.final_act(outputs[-1])
        else:
            final = outputs[-1]
        
        if train:
            return [final] + outputs[:-1]
        
        return final

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create positional encoding matrix once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)

class DeepSleepAttn2DPE(nn.Module):
    def __init__(self, in_channels=9, base_ch=16, num_layers=4,
                 transformer_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        # Encoder
        self.inc = DoubleConv2D(in_channels, base_ch)
        self.downs = nn.ModuleList()
        ch = base_ch
        for _ in range(num_layers):
            self.downs.append(Down2D(ch, ch*2))
            ch *= 2
        
        # Bottleneck dilated conv + SE + Dropout
        self.bot = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            SEBlock(ch),
            nn.Dropout2d(p=dropout)
        )
        
        # Positional Encoding for Transformer
        self.pos_enc = PositionalEncoding(d_model=ch, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=ch, nhead=nhead,
            dim_feedforward=ch*2, dropout=dropout
        )
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        
        # Decoder with deep supervision heads
        self.ups = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(num_layers):
            prev_ch = ch
            skip_ch = prev_ch // 2
            self.ups.append(Up2D(prev_ch, skip_ch))
            self.heads.append(nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(skip_ch, 1, kernel_size=1)
            ))
            ch = skip_ch
        
        self.final_act = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bot(x)  # (B, C, H', W')
        
        B, C, H, W = x.shape
        # Prepare for transformer: flatten H, sequence along W
        x_seq = x.permute(0,2,3,1).reshape(B*H, W, C)  # (B*H, W, C)
        x_seq = x_seq.permute(1,0,2)                   # (W, B*H, C)
        x_seq = self.pos_enc(x_seq)
        x_seq = self.transformer(x_seq)
        x_seq = x_seq.permute(1,0,2).reshape(B, H, W, C).permute(0,3,1,2)
        
        # Decoder with deep supervision
        outputs = []
        for up, head, skip in zip(self.ups, self.heads, reversed(skips[:-1])):
            x_seq = up(x_seq, skip)
            out = head(x_seq)

            if out.shape[2] != H or out.shape[3] != W:
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                out_inter = self.final_act(out_inter)
            outputs.append(out_inter)
        
        final = self.final_act(outputs[-1])
        return final


# Example instantiation
if __name__ == "__main__":
    model = DeepSleepAttn2D(in_channels=9, base_ch=16, num_layers=4, transformer_layers=2, nhead=4)
    dummy = torch.randn(2,9,51,30000)  # smaller width for test
    outs = model(dummy, True)
    for o in outs:
        print(o.shape)
