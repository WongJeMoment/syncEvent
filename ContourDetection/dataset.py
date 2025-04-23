# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from config import IMAGE_DIR, EDGE_DIR, IMG_SIZE

class EdgeDataset(Dataset):
    def __init__(self):
        self.image_list = sorted([
            f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_DIR, self.image_list[idx])
        edge_path = os.path.join(EDGE_DIR, self.image_list[idx])

        image = cv2.imread(img_path)
        edge = cv2.imread(edge_path, 0)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        edge = cv2.resize(edge, (IMG_SIZE, IMG_SIZE))

        image = image.astype(np.float32) / 255.0
        edge = (edge > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)  # [C,H,W]
        edge = torch.from_numpy(edge).unsqueeze(0)        # [1,H,W]

        return image, edge
