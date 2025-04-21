# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import *
from dataset import HeatmapDataset, infer_num_keypoints
from model import HeatmapNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 推断关键点数目
    num_keypoints = infer_num_keypoints(TRAIN_LABEL_DIR)

    dataset = HeatmapDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HeatmapNet(num_keypoints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, heatmaps in loader:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            preds = model(imgs)
            loss = F.mse_loss(preds, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
