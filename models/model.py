import math
import torch
import torch.nn as nn
import timm                 # ✔️ 包含 MobileViTBlock
from einops import rearrange
from timm.models.mobilevit import MobileVitBlock
import torch.nn.functional as F

# Loss Function
def extract_coords_from_heatmap(hm):
    """
    hm: (B, K, H, W)
    return: (B, K, 2) 坐标 (x, y)
    """
    B, K, H, W = hm.shape
    hm = hm.view(B, K, -1)
    coords = hm.argmax(dim=-1)
    x = coords % W
    y = coords // W
    return torch.stack([x, y], dim=-1).float()

def structure_aware_heatmap_loss(pred, target, mask=None, λ1=1.0, λ2=1.0, λ3=1.0):
    """
    综合结构感知损失，包括 heatmap 差异、关键点位置对齐、结构保持
    pred / target: [B, K, H, W]
    mask: optional [B, K, H, W]，用于屏蔽无效通道
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask

    L_heatmap = F.mse_loss(pred, target)

    pred_coords = extract_coords_from_heatmap(pred)
    gt_coords   = extract_coords_from_heatmap(target)
    L_center    = F.l1_loss(pred_coords, gt_coords)

    pred_struct = torch.cdist(pred_coords, pred_coords, p=2)
    gt_struct   = torch.cdist(gt_coords, gt_coords, p=2)
    L_structure = F.l1_loss(pred_struct, gt_struct)

    return λ1 * L_heatmap + λ2 * L_center + λ3 * L_structure


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

class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.ReLU())
        self.up   = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec  = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU())
        self.out  = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d  = self.dec(torch.cat([self.up(e2), e1], 1))
        return self.out(d)

# -------------- Hybrid UNet -------------------- #
class HybridHeatmapUNet(nn.Module):
    def __init__(self, num_keypoints, width_mult=0.75):
        super().__init__()
        ch = [int(c*width_mult) for c in (32, 64, 96, 192, 256)]
        c1, c2, c3, c4, c5 = ch
        self.num_keypoints = num_keypoints

        self.enc1 = DSConvBlock(3,  c1)
        self.pool1= nn.MaxPool2d(2)
        self.enc2 = DSConvBlock(c1, c2)
        self.pool2= nn.MaxPool2d(2)
        self.enc3 = DSConvBlock(c2, c3)
        self.pool3= nn.MaxPool2d(2)
        self.enc4 = DSConvBlock(c3, c4)
        self.pool4= nn.MaxPool2d(2)

        self.mv1 = MobileViT(in_chs=c4, d_model=c4, patch_size=2)
        self.mv2 = MobileViT(in_chs=c4, d_model=c4, patch_size=2)

        self.up4  = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = DSConvBlock(c4 + c4, c4)
        self.up3  = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = DSConvBlock(c3 + c3, c3)
        self.up2  = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DSConvBlock(c2 + c2, c2)
        self.up1  = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DSConvBlock(c1 + c1, c1)

        self.final = nn.Conv2d(c1, num_keypoints, 1)
        self.out_channels = num_keypoints

        # 添加 refinement 模块
        self.refine_net = TinyUNet(in_ch=1)

    def forward(self, x):
        B, _, H, W = x.shape

        # 编码-解码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.pool4(e4)
        b  = self.mv2(self.mv1(b))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        coarse_hm = self.final(d1)

        # ---------- Adaptive Patch Refinement ----------
        refined_hm = coarse_hm.clone()
        patch_size = 32
        pad = patch_size // 2
        threshold = 0.3

        for b in range(B):
            for k in range(self.num_keypoints):
                heat = coarse_hm[b, k]
                maxval = heat.max().item()
                if maxval < threshold:
                    continue
                y, x = torch.nonzero(heat == maxval, as_tuple=True)
                if len(x) == 0: continue
                cy, cx = y[0].item(), x[0].item()
                top = max(cy - pad, 0)
                left = max(cx - pad, 0)
                bottom = min(cy + pad, H)
                right = min(cx + pad, W)

                crop = heat[top:bottom, left:right].unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
                crop_resized = F.interpolate(crop, size=(32, 32), mode='bilinear', align_corners=False)
                refined = self.refine_net(crop_resized)
                refined_up = F.interpolate(refined, size=(bottom - top, right - left), mode='bilinear', align_corners=False)
                refined_hm[b, k, top:bottom, left:right] = refined_up.squeeze(0).squeeze(0)

        return refined_hm