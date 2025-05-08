import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HeatmapUNet  # 或 HeatmapNet
from config import VAL_IMAGE_DIR


def load_image(img_path):
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C, H, W
    return img_tensor.unsqueeze(0), (orig_w, orig_h), img  # (1, C, H, W), 原始尺寸, 原始图像


def extract_peak_coords(heatmap_tensor):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    coords = []
    for hm in heatmap_np:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords.append((x, y))
    return coords


def visualize_prediction(orig_img, pred_heatmap):
    pred_vis = pred_heatmap.squeeze(0).sum(dim=0).detach().cpu().numpy()
    pred_coords = extract_peak_coords(pred_heatmap)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Predicted Heatmap")
    plt.imshow(pred_vis, cmap='hot')
    for i, (x, y) in enumerate(pred_coords):
        plt.scatter(x, y, color='lime', s=30)
        plt.text(x + 2, y - 2, str(i), color='lime', fontsize=8)

    plt.tight_layout()
    plt.show()


def predict_images(model_path="checkpoints/best_model_part2.pt", image_dir=VAL_IMAGE_DIR, num_keypoints=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapUNet(num_keypoints=num_keypoints).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    with torch.no_grad():
        for img_path in image_paths:
            img_tensor, (w, h), orig_img = load_image(img_path)
            img_tensor = img_tensor.to(device)

            pred = model(img_tensor)
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)

            visualize_prediction(orig_img, pred.cpu())


if __name__ == "__main__":
    predict_images()
