import torch
import torch.nn as nn
import torch.nn.functional as F

class CVTAModule(nn.Module):
    def __init__(self, dim=64, depth=3, heads=4):
        super().__init__()
        self.embed = nn.Linear(4, dim)  # 输入：(x, y, Δt, p)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=dim * 2, batch_first=True),
            num_layers=depth
        )
        self.head = nn.Linear(dim, 3)  # 输出 Δu, Δv, disp

    def forward(self, kew_L, kew_R):
        """
        kew_L, kew_R: List[Tensor[M_i, 4]] for each keypoint
        return: delta_uvs, disps
        """
        batch_input = []
        for evL, evR in zip(kew_L, kew_R):
            evL = F.pad(evL, (0, 0, 0, max(0, 256 - evL.shape[0])))  # pad to fixed size
            evR = F.pad(evR, (0, 0, 0, max(0, 256 - evR.shape[0])))
            x = torch.cat([evL, evR], dim=0)  # [2*M, 4]
            batch_input.append(x[:256])
        batch_tensor = torch.stack(batch_input, dim=0)  # [B, 256, 4]

        x_embed = self.embed(batch_tensor)
        encoded = self.encoder(x_embed)
        pred = self.head(encoded.mean(dim=1))  # [B, 3]
        delta_uv = pred[:, :2]  # Δu, Δv
        disp = pred[:, 2]       # disparity
        return delta_uv.tolist(), disp.tolist()
