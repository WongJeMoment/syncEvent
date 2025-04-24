import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import EdgeDataset
from model import EdgeNet
from config import BATCH_SIZE, EPOCHS, LR, CHECKPOINT_PATH


def plot_sample_output(images, preds, edges, epoch, batch_idx):
    """Plot original image, predicted edges, and ground truth edges"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Convert the image tensor to HWC for visualization and plot original image
    axs[0].imshow(images[0].cpu().permute(1, 2, 0))  # Convert from CHW to HWC
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Remove the batch dimension and plot predicted edges
    axs[1].imshow(preds[0].cpu().detach().squeeze(), cmap='gray')  # Squeeze to remove the batch dimension
    axs[1].set_title('Predicted Edges')
    axs[1].axis('off')

    # Remove the batch dimension and plot ground truth edges
    axs[2].imshow(edges[0].cpu().detach().squeeze(), cmap='gray')  # Squeeze to remove the batch dimension
    axs[2].set_title('Ground Truth Edges')
    axs[2].axis('off')

    plt.suptitle(f'Epoch {epoch + 1}, Batch {batch_idx + 1}')
    plt.show()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    dataset = EdgeDataset()
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True)  # Set batch size to 1 for processing one image at a time

    model = EdgeNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, edges) in enumerate(dataloader):
            images, edges = images.to(device), edges.to(device)

            preds = model(images)
            loss = criterion(preds, edges)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Display training results after every 10th batch
            if batch_idx % 10 == 0:
                plot_sample_output(images, preds, edges, epoch, batch_idx)

        avg_loss = total_loss / len(dataloader)
        print(f"🧠 Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"✅ Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
