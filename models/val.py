import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import *
from dataset import HeatmapDataset
from model import HeatmapUNet  # 或 HeatmapNet
import matplotlib.pyplot as plt
import numpy as np
import cv2

def custom_collate_fn(batch):
    imgs, heatmaps = zip(*batch)
    return list(imgs), list(heatmaps)


def validate(model_path="checkpoints/best_model.pt", visualize_sample=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = HeatmapUNet(num_keypoints=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_paths = sorted([
        os.path.join(VAL_IMAGE_DIR, f) for f in os.listdir(VAL_IMAGE_DIR)
        if f.endswith((".jpg", ".png"))
    ])
    label_paths = sorted([
        os.path.join(VAL_LABEL_DIR, os.path.splitext(f)[0] + ".npy")
        for f in os.listdir(VAL_IMAGE_DIR) if f.endswith((".jpg", ".png"))
    ])

    total_loss = 0
    total_dist = 0
    total_keypoints = 0

    os.makedirs("visual_results", exist_ok=True)

    with torch.no_grad():
        for i, (img_path, heatmap_path) in enumerate(zip(image_paths, label_paths)):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C, H, W
            img = img.unsqueeze(0).to(device)  # B, C, H, W

            heatmap = np.load(heatmap_path)
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).to(device)  # B, K, H, W

            pred = model(img)
            pred = match_channels(pred, heatmap)

            loss = F.mse_loss(pred, heatmap)
            total_loss += loss.item()

            acc, dist = accuracy_from_heatmaps(pred, heatmap)
            total_dist += dist
            total_keypoints += acc

            if visualize_sample:
                save_path = os.path.join("visual_results", os.path.splitext(os.path.basename(img_path))[0] + ".png")
                visualize(img[0].cpu(), heatmap[0].cpu(), pred[0].cpu(), save_path)

    avg_loss = total_loss / len(image_paths)
    avg_dist = total_dist / (total_keypoints + 1e-6)

    print(f"🔍 Validation Results:")
    print(f"  Avg Loss     : {avg_loss:.4f}")
    print(f"  Avg Distance : {avg_dist:.2f}")


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


def visualize(img, gt_heatmap, pred_heatmap, save_path=None):
    img = img.permute(1, 2, 0).cpu().numpy()
    gt_vis = gt_heatmap.sum(dim=0).detach().cpu().numpy()
    pred_vis = pred_heatmap.sum(dim=0).detach().cpu().numpy()

    pred_coords = extract_peak_coords(pred_heatmap.unsqueeze(0))  # B, K, H, W → list of (x, y)

    plt.figure(figsize=(12, 4))

    # 原图
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)

    # GT 热图
    plt.subplot(1, 3, 2)
    plt.title("GT Heatmap")
    plt.imshow(gt_vis, cmap='hot')

    # Pred 热图 + 标注点
    plt.subplot(1, 3, 3)
    plt.title("Pred Heatmap")
    plt.imshow(pred_vis, cmap='hot')

    for i, (x, y) in enumerate(pred_coords):
        plt.scatter(x, y, color='lime', s=30)
        plt.text(x + 2, y - 2, str(i), color='lime', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved visualization with keypoints: {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 改为 True 查看第一张可视化样本
    validate(model_path="checkpoints/best_model.pt", visualize_sample=True)
