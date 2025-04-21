# dataset.py
import os, json
import torch
import cv2
from torch.utils.data import Dataset
from config import IMG_SIZE, SIGMA
from utils import generate_heatmap

class KeypointDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.json_paths = sorted([os.path.join(json_dir, f) for f in os.listdir(json_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        with open(self.json_paths[idx]) as f:
            keypoints = np.array(json.load(f), dtype=np.float32)

        heatmaps = generate_heatmap(keypoints, IMG_SIZE, SIGMA)
        return torch.tensor(img), torch.tensor(heatmaps)
