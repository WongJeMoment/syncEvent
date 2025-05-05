import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import *
from dataset import HeatmapDataset
from model import HeatmapUNet  # or HeatmapNet
import matplotlib.pyplot as plt
import numpy as np


def custom_collate_fn(batch):
    # 会将 batch 拆成两个 tuple
    imgs, heatmaps = zip(*batch)
    return list(imgs), list(heatmaps)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HeatmapDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    model = HeatmapUNet(num_keypoints=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    os.makedirs("checkpoints", exist_ok=True)
    best_dist = float('inf')
    best_model_path = "checkpoints/best_model.pt"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_dist = 0
        total_keypoints = 0

        for imgs, heatmaps in loader:
            for img, heatmap in zip(imgs, heatmaps):
                img = img.to(device).unsqueeze(0)
                heatmap = heatmap.to(device).unsqueeze(0)

                pred = model(img)
                pred = match_channels(pred, heatmap)

                pred = F.interpolate(pred, size=heatmap.shape[-2:], mode='bilinear', align_corners=False)

                loss = F.mse_loss(pred, heatmap)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                acc, dist = accuracy_from_heatmaps(pred, heatmap)
                total_dist += dist
                total_keypoints += acc

        avg_dist = total_dist / (total_keypoints + 1e-6)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Avg Distance: {avg_dist:.2f}")

        # 可视化最后一个样本
        visualize(img[0].cpu(), heatmap[0].cpu(), pred[0].cpu())

        # ✅ 只保留当前最佳模型，删除旧的
        if avg_dist < best_dist:
            best_dist = avg_dist
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} (Avg Dist: {avg_dist:.2f})")


def match_channels(pred, heatmap):
    _, c_pred, h, w = pred.shape
    _, c_gt, _, _ = heatmap.shape
    if c_pred == c_gt:
        return pred
    elif c_pred > c_gt:
        return pred[:, :c_gt, :, :]
    else:
        pad = torch.zeros((1, c_gt - c_pred, h, w), device=pred.device, dtype=pred.dtype)
        return torch.cat([pred, pad], dim=1)


def accuracy_from_heatmaps(pred, gt):
    pred_coords = extract_peak_coords(pred)
    gt_coords = extract_peak_coords(gt)
    dist = 0.0
    count = min(len(pred_coords), len(gt_coords))
    for i in range(count):
        px, py = pred_coords[i]
        gx, gy = gt_coords[i]
        dist += np.linalg.norm([px - gx, py - gy])
    return count, dist


def extract_peak_coords(heatmap_tensor):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    coords = []
    for hm in heatmap_np:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords.append((x, y))
    return coords


def visualize(img, gt_heatmap, pred_heatmap):
    img = img.permute(1, 2, 0).cpu().numpy()
    gt_vis = gt_heatmap.sum(dim=0).detach().cpu().numpy()
    pred_vis = pred_heatmap.sum(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("GT Heatmap")
    plt.imshow(gt_vis, cmap='hot')

    plt.subplot(1, 3, 3)
    plt.title("Pred Heatmap")
    plt.imshow(pred_vis, cmap='hot')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
