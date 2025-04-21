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
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        heatmap = np.load(self.heatmap_paths[idx])  # shape: [K, H, W]

        return torch.tensor(img), torch.tensor(heatmap)
