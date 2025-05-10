# dataset.py
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import IMG_SIZE

def infer_num_keypoints(heatmap_dir):
    sample_file = sorted([f for f in os.listdir(heatmap_dir) if f.endswith('.npy')])[0]
    heatmap = np.load(os.path.join(heatmap_dir, sample_file))
    return heatmap.shape[0]  # channel 数 == 关键点数量

def preprocess_image(img, img_size=IMG_SIZE):
    h_orig, w_orig = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_chw = img_normalized.transpose(2, 0, 1).astype(np.float32)

    scale_x = w_orig / img_size
    scale_y = h_orig / img_size

    return img_chw, (scale_x, scale_y), (h_orig, w_orig)



class HeatmapDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
        ])
        self.heatmap_paths = sorted([
            os.path.join(heatmap_dir, os.path.splitext(os.path.basename(f))[0] + '.npy')
            for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        heatmap = np.load(self.heatmap_paths[idx])  # shape: [K, H, W]

        # 自动获取 heatmap 尺寸
        h, w = heatmap.shape[1], heatmap.shape[2]
        img = cv2.resize(img, (w, h))  # 注意: OpenCV 的顺序是 (width, height)

        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        return torch.tensor(img), torch.tensor(heatmap)

