from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import os

class HeatmapDataset(Dataset):
    def __init__(self, img_dir, heatmap_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.heatmap_paths = sorted([os.path.join(heatmap_dir, f.replace('.jpg', '.npy')) for f in os.listdir(img_dir)])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(img, (256, 256))
        img = self.transform(img)

        heatmap = np.load(self.heatmap_paths[idx])
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return img, heatmap
