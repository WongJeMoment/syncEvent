import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import EdgeDataset
from model import EdgeNet
from config import CHECKPOINT_PATH
import cv2
import numpy as np

# ------------------ Canny Edge Detection ------------------
def canny_edge_detection(image):
    """Applies Canny edge detection to the input image tensor (C, H, W)."""
    image = (image * 255).byte().permute(1, 2, 0).numpy()  # Tensor [C,H,W] -> [H,W,C], 0~255 uint8
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges = edges.astype(np.float32) / 255.0  # Normalize
    edges = np.expand_dims(edges, axis=0)  # (H,W) -> (1,H,W)
    return torch.tensor(edges, dtype=torch.float32)

# ------------------ Plot Function ------------------
def plot_validation_results(images, preds, ground_truth, batch_idx):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    img = images[0].cpu().permute(1, 2, 0).numpy()
    axs[0].imshow((img * 255).astype(np.uint8))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Predicted Edge Map
    axs[1].imshow(preds[0].cpu().detach().squeeze(), cmap='gray')
    axs[1].set_title('Predicted Edges')
    axs[1].axis('off')

    # Ground Truth Canny
    axs[2].imshow(ground_truth[0].squeeze().cpu(), cmap='gray')
    axs[2].set_title('Ground Truth (Canny)')
    axs[2].axis('off')

    plt.suptitle(f'Validation - Batch {batch_idx}')
    plt.tight_layout()
    plt.show()

# ------------------ Validation Loop ------------------
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            images = images.cuda()

            preds = model(images)

            ground_truth = []
            for img in images.cpu():  # Already in [C,H,W], float
                edge = canny_edge_detection(img)
                ground_truth.append(edge)

            ground_truth = torch.stack(ground_truth).cuda()

            loss = criterion(preds, ground_truth)
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                plot_validation_results(images, preds, ground_truth, batch_idx)

    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

# ------------------ Main Entry ------------------
def main():
    model = EdgeNet().cuda()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("Model loaded successfully!")

    criterion = torch.nn.BCELoss()

    val_dataset = EdgeDataset()  # ✅ 使用 config 中硬编码路径
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    validate(model, val_loader, criterion)

if __name__ == "__main__":
    main()
