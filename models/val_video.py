import os
import torch
import cv2
import numpy as np
from model import HeatmapUNet  # 或 HeatmapNet
from config import *
from dataset import preprocess_image
import matplotlib.pyplot as plt
import scipy.ndimage

class KalmanPoint:
    def __init__(self):
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # 初始协方差
        self.F = np.eye(4)  # 状态转移矩阵
        self.F[0, 2] = 1
        self.F[1, 3] = 1
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # 观测矩阵
        self.R = np.eye(2) * 25  # 测量噪声
        self.Q = np.eye(4) * 0.01  # 过程噪声

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_pos(self):
        return int(self.state[0]), int(self.state[1])

    def set_pos(self, x, y):
        self.state[0] = x
        self.state[1] = y
        self.state[2] = 0
        self.state[3] = 0

class KalmanSmoother:
    def __init__(self):
        self.trackers = []

    def update(self, keypoints):
        if len(self.trackers) != len(keypoints):
            self.trackers = [KalmanPoint() for _ in keypoints]
            for tracker, (x, y) in zip(self.trackers, keypoints):
                tracker.set_pos(x, y)

        smoothed = []
        for tracker, (x, y) in zip(self.trackers, keypoints):
            tracker.predict()
            tracker.update(np.array([x, y]))
            smoothed.append(tracker.get_pos())

        return smoothed

# 通道匹配--让 pred 和 heatmap 的通道数对齐
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

# 从热图中提取关键点坐标
def extract_peak_coords(heatmap_tensor, orig_size=None, threshold=0.15, nms_radius=5, merge_distance=15):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    coords = []

    for hm in heatmap_np:
        neighborhood = (nms_radius * 2) + 1
        local_max = (hm == scipy.ndimage.maximum_filter(hm, size=neighborhood))
        mask = (hm > threshold) & local_max

        ys, xs = np.where(mask)
        keypoints = [(x, y, hm[y, x]) for x, y in zip(xs, ys)]

        keypoints.sort(key=lambda k: k[2], reverse=True)

        selected = []
        for x, y, score in keypoints:
            too_close = False
            for sel_x, sel_y in selected:
                if np.hypot(x - sel_x, y - sel_y) < merge_distance:
                    too_close = True
                    break
            if not too_close:
                selected.append((x, y))

        for (x, y) in selected:
            if orig_size is not None:
                scale_x = orig_size[1] / hm.shape[1]
                scale_y = orig_size[0] / hm.shape[0]
                x = int(x * scale_x)
                y = int(y * scale_y)
            coords.append((x, y))

    return coords

# 关键点平滑器
# class KeypointSmoother:
#     def __init__(self, window_size=5):
#         self.window_size = window_size
#         self.buffers = None  # 每个点有自己的小buffer列表
#
#     def update(self, keypoints):
#         keypoints = np.array(keypoints)
#         if self.buffers is None:
#             # 第一次初始化
#             self.buffers = [[] for _ in range(len(keypoints))]
#
#         if len(keypoints) != len(self.buffers):
#             # 数量对不上，重新初始化
#             self.buffers = [[] for _ in range(len(keypoints))]
#
#         for i, (x, y) in enumerate(keypoints):
#             self.buffers[i].append((x, y))
#             if len(self.buffers[i]) > self.window_size:
#                 self.buffers[i].pop(0)
#
#         return self.get_smoothed()
#
#     def get_smoothed(self):
#         smoothed = []
#         for buffer in self.buffers:
#             if buffer:
#                 avg_x = np.mean([p[0] for p in buffer])
#                 avg_y = np.mean([p[1] for p in buffer])
#                 smoothed.append((int(avg_x), int(avg_y)))
#         return smoothed

# 标注关键点
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

    smoother = KalmanSmoother()  # 初始化平滑器

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🚪 视频结束，优雅退出。")
            break

        orig_h, orig_w = frame.shape[:2]
        img_input = preprocess_image(frame)
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_heatmap = model(img_tensor)

        keypoints = extract_peak_coords(pred_heatmap, orig_size=(orig_h, orig_w))
        keypoints = smoother.update(keypoints)  # 加入平滑！

        vis_frame = visualize_keypoints(frame.copy(), keypoints)

        cv2.imshow("Keypoint Detection", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 手动退出。")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/wangzhe/ICRA2025/MY/demo/left.mp4"
    val_video(video_path)
