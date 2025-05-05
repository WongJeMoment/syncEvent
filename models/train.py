import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import *
from dataset import HeatmapDataset
from model import HeatmapUNet


def scan_max_keypoints(label_dir):
    npy_files = glob.glob(os.path.join(label_dir, "*.npy"))
    if not npy_files:
        raise ValueError("No .npy files found in label directory.")
    return max(np.load(f, mmap_mode="r").shape[0] for f in npy_files)


def match_channels(pred, k):
    # pred shape: (1, C_pred, H, W)
    c_pred = pred.size(1)
    if c_pred == k:
        return pred
    elif c_pred > k:
        return pred[:, :k]
    else:
        pad = torch.zeros((1, k - c_pred, *pred.shape[2:]),
                          device=pred.device, dtype=pred.dtype)
        return torch.cat([pred, pad], dim=1)


def extract_peak_coords(hm_tensor):
    hm_np = hm_tensor.squeeze(0).detach().cpu().numpy()
    return [tuple(np.unravel_index(np.argmax(hm), hm.shape)[::-1]) for hm in hm_np]


def accuracy_from_heatmaps(pred, gt):
    pred_coords = extract_peak_coords(pred)
    gt_coords = extract_peak_coords(gt)
    dist = sum(np.linalg.norm(np.subtract(p, g)) for p, g in zip(pred_coords, gt_coords))
    return len(gt_coords), dist


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


def custom_collate_fn(batch):
    imgs, heatmaps = zip(*batch)
    return list(imgs), list(heatmaps)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HeatmapDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=custom_collate_fn)

    max_keypoints = scan_max_keypoints(TRAIN_LABEL_DIR)
    print(f"✅ 模型最大输出关键点数设置为: {max_keypoints}")
    model = HeatmapUNet(num_keypoints=max_keypoints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dist = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = "checkpoints/best_model.pt"

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_dist, total_kpts = 0, 0, 0
        epoch_max_kps = 0

        for imgs, heatmaps in loader:
            for img, heatmap in zip(imgs, heatmaps):
                img = img.to(device).unsqueeze(0)
                heatmap = heatmap.to(device).unsqueeze(0)
                k = heatmap.shape[1]  # 当前图实际关键点数
                epoch_max_kps = max(epoch_max_kps, k)

                pred = model(img)
                pred = match_channels(pred, k)
                pred = F.interpolate(pred, size=heatmap.shape[-2:], mode="bilinear", align_corners=False)

                loss = F.mse_loss(pred, heatmap)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                acc, dist = accuracy_from_heatmaps(pred, heatmap)
                total_kpts += acc
                total_dist += dist

        avg_dist = total_dist / (total_kpts + 1e-6)
        print(f"▶ Detected max keypoints = {epoch_max_kps}")
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Avg Distance: {avg_dist:.2f}")

        visualize(img[0].cpu(), heatmap[0].cpu(), pred[0].cpu())

        if avg_dist < best_dist:
            best_dist = avg_dist
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} (Avg Dist: {avg_dist:.2f})")


if __name__ == "__main__":
    train()
