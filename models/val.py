import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import IMG_SIZE, VAL_IMAGE_DIR
from model import HybridHeatmapUNet  # 或 HeatmapNet


def load_image(img_path):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)  # B, C, H, W


def extract_peak_coords(heatmap_tensor):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    coords = []
    for hm in heatmap_np:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords.append((x, y))
    return coords


def visualize_prediction(img_tensor, pred_heatmap):
    img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_vis = pred_heatmap.squeeze(0).sum(dim=0).detach().cpu().numpy()
    pred_coords = extract_peak_coords(pred_heatmap)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Predicted Keypoints")
    plt.imshow(pred_vis, cmap='hot')
    for i, (x, y) in enumerate(pred_coords):
        plt.scatter(x, y, color='lime', s=30)
        plt.text(x + 2, y - 2, str(i), color='lime', fontsize=8)

    plt.tight_layout()
    plt.show()


def predict_images(model_path="checkpoints/best_model.pt", image_dir=VAL_IMAGE_DIR, num_keypoints=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridHeatmapUNet(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    with torch.no_grad():
        for img_path in image_paths:
            img_tensor = load_image(img_path).to(device)
            pred = model(img_tensor)
            visualize_prediction(img_tensor.cpu(), pred.cpu())


if __name__ == "__main__":
    predict_images()
