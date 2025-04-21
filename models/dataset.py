# dataset.py
import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import IMG_SIZE, SIGMA
from utils import generate_heatmap

def infer_num_keypoints(json_dir):
    sample_file = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])[0]
    with open(os.path.join(json_dir, sample_file)) as f:
        keypoints = json.load(f)
    return len(keypoints)

class KeypointDataset(Dataset):
    def __init__(self, image_dir, json_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
        ])
        self.json_paths = sorted([
            os.path.join(json_dir, os.path.splitext(os.path.basename(f))[0] + '.json')
            for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        with open(self.json_paths[idx]) as f:
            keypoints = np.array(json.load(f), dtype=np.float32)

        # 坐标按原图分辨率缩放
        scale_x = IMG_SIZE / w0
        scale_y = IMG_SIZE / h0
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        heatmaps = generate_heatmap(keypoints, IMG_SIZE, SIGMA)

        return torch.tensor(img), torch.tensor(heatmaps)
