import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from config import *
from dataset import HeatmapDataset
from model import HybridHeatmapUNet

# 统计最大关键点数
def scan_all_keypoints(label_dir):
    """
    扫描所有 .npy 标签文件，返回最大关键点数量
    """
    npy_files = glob.glob(os.path.join(label_dir, "*.npy"))
    if not npy_files:
        raise ValueError("No .npy files found in label directory.")
    return max(np.load(f, mmap_mode="r").shape[0] for f in npy_files)

# 提取预测/GT热图中的关键点坐标
def extract_peak_coords(hm_tensor):
    hm_np = hm_tensor.detach().cpu().numpy()
    coords = []
    # 遍历每个样本的热图
    for hm in hm_np:
        # 找到每个热图的最大值坐标
        coords.append([tuple(np.unravel_index(np.argmax(h, axis=None), h.shape)[::-1]) for h in hm])
    return coords

# 提取预测/GT热图中的关键点坐标
def accuracy_from_heatmaps(pred, gt):
    # 代表模型预测输出的热图-->模型输出热图
    pred_coords = extract_peak_coords(pred)
    # 标签热图-->标签输出热图
    gt_coords = extract_peak_coords(gt)
    dist = 0
    count = 0
    # pc, gc 是每张图片预测和GT关键点坐标的列表
    for pc, gc in zip(pred_coords, gt_coords):
        # p, g 是单个关键点坐标，形式为 (x, y)
        for p, g in zip(pc, gc):
            # 计算欧式距离
            dist += np.linalg.norm(np.subtract(p, g))
            # 记录关键点总数，用于后续平均距离计算
            count += 1
    # count: 所有关键点总数
    # dist: 所有关键点之间的总欧氏距离
    return count, dist

# 三图对比可视化
def visualize(img, gt_heatmap, pred_heatmap):
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    gt_vis = gt_heatmap.sum(0).detach().cpu().numpy()
    pred_vis = pred_heatmap.sum(0).detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    titles = ["Image", "GT Heatmap", "Pred Heatmap"]
    for i, data in enumerate([img, gt_vis, pred_vis]):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        plt.imshow(data if i == 0 else data, cmap='hot')
    plt.tight_layout()
    plt.show()

# DataLoader打包
def custom_collate_fn(batch):
    imgs, heatmaps = zip(*batch)
    return list(imgs), list(heatmaps)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设备与数据加载
    dataset = HeatmapDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=custom_collate_fn)
    # 动态设置关键点通道数
    max_keypoints = scan_all_keypoints(TRAIN_LABEL_DIR)
    print(f"✅ 模型最大输出关键点数设置为: {max_keypoints}")
    model = HybridHeatmapUNet(num_keypoints=max_keypoints).to(device)
    # 最多关键点数
    model.out_channels = max_keypoints  # ✅ 添加属性用于检查输出通道数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dist = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = "checkpoints/best_model.pt"

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_dist, total_kpts = 0, 0, 0

        for imgs, heatmaps in loader:
            imgs = torch.stack(imgs).to(device)         # (B, 3, H, W)
            heatmaps = torch.stack(heatmaps).to(device) # (B, K, H, W)

            # 关键点热图通道对齐：裁剪或补零
            if heatmaps.shape[1] > model.out_channels:
                heatmaps = heatmaps[:, :model.out_channels]
            elif heatmaps.shape[1] < model.out_channels:
                pad = model.out_channels - heatmaps.shape[1]
                padding = torch.zeros((heatmaps.shape[0], pad, *heatmaps.shape[2:]),
                                      device=heatmaps.device, dtype=heatmaps.dtype)
                heatmaps = torch.cat([heatmaps, padding], dim=1)

            k = heatmaps.shape[1]  # == model.out_channels
            # 热图预测和 loss 计算（使用 mask 筛选有效关键点）
            preds = model(imgs)
            # 被上采样为 (4, 15, 128, 128)，与 heatmaps 对齐
            preds = F.interpolate(preds, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False)

            # ✅ 用 mask 区分有效关键点通道
            mask = torch.zeros_like(preds)
            mask[:, :k] = 1.0
            loss = F.mse_loss(preds * mask, heatmaps * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ 准确度评估仅对有效通道
            acc, dist = accuracy_from_heatmaps(preds[:, :k], heatmaps[:, :k])
            total_kpts += acc
            total_dist += dist

        avg_dist = total_dist / (total_kpts + 1e-6)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Avg Distance: {avg_dist:.2f}")

        visualize(imgs[0].cpu(), heatmaps[0].cpu(), preds[0].cpu())

        if avg_dist < best_dist:
            best_dist = avg_dist
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} (Avg Dist: {avg_dist:.2f})")


if __name__ == "__main__":
    train()
