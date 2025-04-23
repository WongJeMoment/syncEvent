# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import EdgeDataset
from model import EdgeNet
from config import BATCH_SIZE, EPOCHS, LR, CHECKPOINT_PATH

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    dataset = EdgeDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EdgeNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, edges in dataloader:
            images, edges = images.to(device), edges.to(device)

            preds = model(images)
            loss = criterion(preds, edges)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"🧠 Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"✅ Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
