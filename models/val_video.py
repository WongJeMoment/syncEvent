import os
import torch
import cv2
import numpy as np
import scipy.ndimage
from model import HeatmapUNet  # or HeatmapNet
from config import *
from dataset import preprocess_image
import matplotlib.pyplot as plt

# --------- 卡尔曼滤波单点追踪器 ---------
class KalmanPoint:
    def __init__(self):
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000
        self.F = np.eye(4)
        self.F[0, 2] = 1
        self.F[1, 3] = 1
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 25
        self.Q = np.eye(4) * 0.01

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

# --------- 多角点追踪器 ---------
class MultiPointTracker:
    def __init__(self, max_distance=30):
        self.trackers = []
        self.max_distance = max_distance

    def init_trackers(self, keypoints):
        self.trackers = [KalmanPoint() for _ in keypoints]
        for tracker, (x, y) in zip(self.trackers, keypoints):
            tracker.set_pos(x, y)

    def predict(self):
        for tracker in self.trackers:
            tracker.predict()

    def update(self, detections):
        updated_flags = [False] * len(self.trackers)
        used = [False] * len(detections)

        for i, tracker in enumerate(self.trackers):
            pred_x, pred_y = tracker.get_pos()

            best_dist = self.max_distance
            best_j = -1

            for j, (x, y) in enumerate(detections):
                if used[j]:
                    continue
                dist = np.hypot(pred_x - x, pred_y - y)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                tracker.update(np.array([detections[best_j][0], detections[best_j][1]]))
                used[best_j] = True
                updated_flags[i] = True

        return [tracker.get_pos() for tracker in self.trackers]

# --------- 角点提取函数 ---------
def extract_peak_coords(heatmap_tensor, orig_size=None, threshold=0.15, nms_radius=5, merge_distance=15, top_k=8):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    raw_coords = []

    for hm in heatmap_np:
        neighborhood = (nms_radius * 2) + 1
        local_max = (hm == scipy.ndimage.maximum_filter(hm, size=neighborhood))
        mask = (hm > threshold) & local_max

        ys, xs = np.where(mask)
        keypoints = [(x, y, hm[y, x]) for x, y in zip(xs, ys)]

        for (x, y, score) in keypoints:
            if orig_size is not None:
                scale_x = orig_size[1] / hm.shape[1]
                scale_y = orig_size[0] / hm.shape[0]
                x = int(x * scale_x)
                y = int(y * scale_y)
            raw_coords.append((x, y, score))

    raw_coords.sort(key=lambda k: k[2], reverse=True)

    final_coords = []
    visited = np.zeros(len(raw_coords), dtype=bool)

    for i, (x_i, y_i, score_i) in enumerate(raw_coords):
        if visited[i]:
            continue
        final_coords.append((x_i, y_i, score_i))
        for j, (x_j, y_j, _) in enumerate(raw_coords):
            if not visited[j]:
                if np.hypot(x_i - x_j, y_i - y_j) < merge_distance:
                    visited[j] = True

    if len(final_coords) > top_k:
        final_coords = final_coords[:top_k]

    return [(int(x), int(y)) for (x, y, _) in final_coords]

# --------- 可视化函数 ---------
def visualize_keypoints(img, keypoints):
    for x, y in keypoints:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    return img

# --------- 主程序 ---------
def val_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeatmapUNet(num_keypoints=8).to(device)
    best_model_path = "checkpoints/best_model.pt"
    assert os.path.exists(best_model_path), "没有找到最佳模型，请先训练。"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return

    tracker = None
    first_frame = True

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

        if first_frame:
            tracker = MultiPointTracker(max_distance=30)
            tracker.init_trackers(keypoints)
            first_frame = False
        else:
            tracker.predict()
            keypoints = tracker.update(keypoints)

        vis_frame = visualize_keypoints(frame.copy(), keypoints)

        cv2.imshow("Keypoint Detection and Tracking", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 手动退出。")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/wangzhe/ICRA2025/MY/demo/left.mp4"
    val_video(video_path)
