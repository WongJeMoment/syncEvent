import math
import torch
import torch.nn as nn
import timm                 # ✔️ 包含 MobileViTBlock
from einops import rearrange
from timm.models.mobilevit import MobileVitBlock

# ---------------- DSConv + ECA ---------------- #
class ECA(nn.Module):
    def __init__(self, c, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(c) / gamma) + b))
        k = k if k % 2 else k + 1
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)                      # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sig(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_eca=True, groups=8):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.eca  = ECA(out_ch) if use_eca else nn.Identity()

    def forward(self, x):
        x = self.act(self.norm(self.pw(self.dw(x))))
        return self.eca(x)

# -------------- MobileViT Bottleneck ----------- #
def MobileViT(in_chs, d_model=192, patch_size=2):
    return MobileVitBlock(
        in_chs=in_chs,               # 必须提供
        out_chs=d_model,             # 输出通道数
        patch_size=patch_size,
        kernel_size=3,
        stride=1
    )



# -------------- Hybrid UNet -------------------- #
class HybridHeatmapUNet(nn.Module):
    def __init__(self, num_keypoints, width_mult=0.75):
        super().__init__()
        ch = [int(c*width_mult) for c in (32, 64, 96, 192, 256)]  # c1..c5
        c1, c2, c3, c4, c5 = ch

        # 编码器
        self.enc1 = DSConvBlock(3,  c1)
        self.pool1= nn.MaxPool2d(2)
        self.enc2 = DSConvBlock(c1, c2)
        self.pool2= nn.MaxPool2d(2)
        self.enc3 = DSConvBlock(c2, c3)
        self.pool3= nn.MaxPool2d(2)
        self.enc4 = DSConvBlock(c3, c4)
        self.pool4= nn.MaxPool2d(2)

        # Hybrid Bottleneck: 2×MobileViT
        self.mv1 = MobileViT(in_chs=c4, d_model=c4, patch_size=2)
        self.mv2 = MobileViT(in_chs=c4, d_model=c4, patch_size=2)

        # 解码器（仍用 DSConv）
        self.up4  = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = DSConvBlock(c4 + c4, c4)
        self.up3  = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = DSConvBlock(c3 + c3, c3)
        self.up2  = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DSConvBlock(c2 + c2, c2)
        self.up1  = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DSConvBlock(c1 + c1, c1)

        # 输出层
        self.final = nn.Conv2d(c1, num_keypoints, 1)
        self.out_channels = num_keypoints

    def forward(self, x):
        # --- 编码 ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # --- Bottleneck / MobileViT ---
        b  = self.pool4(e4)                 # ↓16
        b  = self.mv2(self.mv1(b))          # global context

        # --- 解码 ---
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.final(d1)               # [B, K, H, W]
