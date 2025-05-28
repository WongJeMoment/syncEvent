import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from thop import profile

from model import HybridHeatmapUNet
from config import VAL_IMAGE_DIR, VAL_HEATMAP_DIR

def load_image(img_path):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)  # (1, 3, H, W)

def load_heatmap(label_path):
    heatmap = np.load(label_path)
    return torch.from_numpy(heatmap).unsqueeze(0).float()  # (1, K, H, W)

def extract_peak_coords(heatmap, threshold=0.01):
    B, K, H, W = heatmap.shape
    coords = []
    for b in range(B):
        sample_coords = []
        for k in range(K):
            hm = heatmap[b, k]
            max_val = hm.max()
            if max_val < threshold:
                sample_coords.append((-1, -1))
                continue
            y, x = torch.nonzero(hm == max_val, as_tuple=True)
            if len(x) == 0:
                sample_coords.append((-1, -1))
            else:
                sample_coords.append((x[0].item(), y[0].item()))
        coords.append(sample_coords)
    return coords

def scale_coords(coords, from_size, to_size):
    scale_x = to_size[1] / from_size[1]
    scale_y = to_size[0] / from_size[0]
    return [(x * scale_x, y * scale_y) if x >= 0 and y >= 0 else (-1, -1) for (x, y) in coords]

def visualize_prediction(img_tensor, pred_heatmap, idx):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_vis = pred_heatmap.squeeze(0).sum(dim=0).detach().cpu().numpy()
    pred_coords = extract_peak_coords(pred_heatmap, threshold=0.01)[0]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Input Image {idx}")
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Keypoints {idx}")
    plt.imshow(pred_vis, cmap='hot')
    for i, (x, y) in enumerate(pred_coords):
        if x >= 0 and y >= 0:
            plt.scatter(x, y, color='lime', s=30)
            plt.text(x + 2, y - 2, str(i), color='lime', fontsize=8)
        else:
            plt.text(10, 20 + 10 * i, f"Missing: {i}", color='red')
    plt.tight_layout()
    plt.show()

def get_model_stats(model, input_size=(1, 3, 256, 256), runs=100):
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    params_m = round(params / 1e6, 2)
    flops_g = round(flops / 1e9, 2)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start = time.time()
        for _ in range(runs):
            _ = model(dummy_input)
        end = time.time()
    fps = round(runs / (end - start), 2)
    return {"Params (M)": params_m, "FLOPs (G)": flops_g, "FPS": fps}

def validate(model_path, image_dir, label_dir, num_keypoints, pck_threshold=0.05, visualize_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridHeatmapUNet(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    stats = get_model_stats(model)
    print("\n✅ 模型信息:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])
    label_paths = sorted([
        os.path.join(label_dir, f) for f in os.listdir(label_dir)
        if f.lower().endswith(".npy")
    ])

    assert len(image_paths) == len(label_paths), "图像与标签数量不一致"

    total_dist = 0
    total_kpts = 0
    correct_kpts = 0
    per_kpt_dist = np.zeros(num_keypoints)
    per_kpt_count = np.zeros(num_keypoints)

    with torch.no_grad():
        for idx, (img_path, label_path) in enumerate(zip(image_paths, label_paths)):
            img_tensor = load_image(img_path).to(device)
            gt_heatmap = load_heatmap(label_path).to(device)
            pred = model(img_tensor)
            pred = F.interpolate(pred, size=gt_heatmap.shape[-2:], mode="bilinear", align_corners=False)

            gt_k = gt_heatmap.shape[1]
            if gt_k < num_keypoints:
                pad = torch.zeros((1, num_keypoints - gt_k, *gt_heatmap.shape[2:]), device=device)
                gt_heatmap = torch.cat([gt_heatmap, pad], dim=1)
            elif gt_k > num_keypoints:
                gt_heatmap = gt_heatmap[:, :num_keypoints]

            H, W = gt_heatmap.shape[-2:]
            norm_radius = max(H, W) * pck_threshold

            pred_coords = extract_peak_coords(pred, threshold=0.01)[0]
            gt_coords = extract_peak_coords(gt_heatmap, threshold=0.01)[0]
            pred_coords = scale_coords(pred_coords, (H, W), (256, 256))
            gt_coords = scale_coords(gt_coords, (H, W), (256, 256))

            for i, (p, g) in enumerate(zip(pred_coords, gt_coords)):
                if g == (-1, -1) or p == (-1, -1):
                    continue
                p = np.array(p)
                g = np.array(g)
                dist = np.linalg.norm(p - g)
                total_dist += dist
                total_kpts += 1
                per_kpt_dist[i] += dist
                per_kpt_count[i] += 1
                if dist <= norm_radius:
                    correct_kpts += 1

            if visualize_idx is not None and idx == visualize_idx:
                visualize_prediction(img_tensor.cpu(), pred.cpu(), idx)

    avg_dist = total_dist / (total_kpts + 1e-6)
    pck = correct_kpts / (total_kpts + 1e-6)

    print(f"\n✅ 验证结果（有效关键点）:")
    print(f"  - 平均误差: {avg_dist:.2f} px")
    print(f"  - PCK@{pck_threshold}: {pck * 100:.2f}%")
    print(f"  - 有效关键点总数: {total_kpts}")

    for k in range(num_keypoints):
        if per_kpt_count[k] > 0:
            print(f"  - 第 {k:2d} 个关键点平均误差: {per_kpt_dist[k] / per_kpt_count[k]:.2f}")
        else:
            print(f"  - 第 {k:2d} 个关键点无有效样本")

if __name__ == "__main__":
    validate(
        model_path="/home/wangzhe/ICRA2025/MY/models/checkpoints/Cube/best_model.pt",
        image_dir=VAL_IMAGE_DIR,
        label_dir=VAL_HEATMAP_DIR,
        num_keypoints=7,
        pck_threshold=0.05,
        visualize_idx=10  # ✅ 可设置为任意图像索引
    )
