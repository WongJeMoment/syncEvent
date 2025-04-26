import os
import torch
import cv2
import numpy as np
from model import HeatmapUNet  # 或 HeatmapNet
from config import *
from dataset import preprocess_image  # 需要你在dataset.py里加一个预处理函数（下面教你）
import matplotlib.pyplot as plt


def match_channels(pred, heatmap):
    _, c_pred, h, w = pred.shape
    _, c_gt, _, _ = heatmap.shape
    if c_pred == c_gt:
        return pred
    elif c_pred > c_gt:
        return pred[:, :c_gt, :, :]
    else:
        pad = torch.zeros((1, c_gt - c_pred, h, w), device=pred.device, dtype=pred.dtype)
        return torch.cat([pred, pad], dim=1)


def extract_peak_coords(heatmap_tensor, orig_size=None):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    coords = []
    for hm in heatmap_np:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        if orig_size is not None:
            scale_x = orig_size[1] / hm.shape[1]
            scale_y = orig_size[0] / hm.shape[0]
            x = int(x * scale_x)
            y = int(y * scale_y)
        coords.append((x, y))
    return coords


def visualize_keypoints(img, keypoints):
    for x, y in keypoints:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    return img


def val_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeatmapUNet(num_keypoints=8).to(device)
    best_model_path = "checkpoints/best_model.pt"
    assert os.path.exists(best_model_path), "没有找到最佳模型，请先训练一遍。"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🚪 视频结束，优雅退出。")
            break

        orig_h, orig_w = frame.shape[:2]  # 👈 保存原图尺寸
        img_input = preprocess_image(frame)
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_heatmap = model(img_tensor)

        # 提取关键点，并映射回原始尺寸
        keypoints = extract_peak_coords(pred_heatmap, orig_size=(orig_h, orig_w))

        vis_frame = visualize_keypoints(frame.copy(), keypoints)

        cv2.imshow("Keypoint Detection", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 手动退出。")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 自己换成你的视频路径～
    video_path = "/home/wangzhe/ICRA2025/MY/demo/left.mp4"
    val_video(video_path)
