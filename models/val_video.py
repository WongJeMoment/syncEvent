import os
import time
import torch
import numpy as np
import cv2
import scipy.ndimage
from model import HybridHeatmapUNet
from config import *

# ---------- 预处理图像并 pad 到 32 的倍数 ----------
def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img

def pad_to_multiple(img, divisor=32):
    h, w = img.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return img_padded, (h, w)


# ---------- 卡尔曼滤波器 ----------
class KalmanPoint:
    def __init__(self):
        self.state = np.zeros(4)
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


# ---------- 多点追踪器 ----------
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


# ---------- 提取关键点 ----------
def extract_peak_coords(heatmap_tensor, threshold=0.15, nms_radius=5, merge_distance=15, top_k=14, orig_size=None):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    raw_coords = []

    for hm in heatmap_np:
        neighborhood = (nms_radius * 2) + 1
        local_max = (hm == scipy.ndimage.maximum_filter(hm, size=neighborhood))
        mask = (hm > threshold) & local_max
        ys, xs = np.where(mask)
        keypoints = [(x, y, hm[y, x]) for x, y in zip(xs, ys)]
        raw_coords.extend(keypoints)

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

    coords = [(x, y) for (x, y, _) in final_coords]

    if orig_size is not None:
        h_hm, w_hm = heatmap_np.shape[1:]
        scale_x = orig_size[1] / w_hm
        scale_y = orig_size[0] / h_hm
        coords = [(int(x * scale_x), int(y * scale_y)) for (x, y) in coords]

    return coords


# ---------- 可视化关键点 ----------
def visualize_keypoints(img, keypoints):
    for idx, (x, y) in enumerate(keypoints):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return img


# ---------- 主函数 ----------
def val_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridHeatmapUNet(num_keypoints=15).to(device)
    model_path = "checkpoints/best_model.pt"
    assert os.path.exists(model_path), "❌ 未找到模型"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return

    tracker = None
    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ 视频结束")
            break

        t_start = time.time()

        # ✅ 统一为 1280x720
        frame_resized = cv2.resize(frame, (1280, 720))

        # ✅ 预处理并 padding
        padded_img, orig_hw = pad_to_multiple(frame_resized)
        img_input = preprocess_image(padded_img)
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_heatmap = model(img_tensor).cpu()
            pred_heatmap = pred_heatmap[:, :, :orig_hw[0], :orig_hw[1]]  # 剪裁回原始尺寸

        keypoints = extract_peak_coords(pred_heatmap, orig_size=orig_hw)

        if first_frame:
            tracker = MultiPointTracker()
            tracker.init_trackers(keypoints)
            first_frame = False
        else:
            tracker.predict()
            keypoints = tracker.update(keypoints)

        vis_frame = visualize_keypoints(frame_resized.copy(), keypoints)

        # ✅ 显示帧率 + 保持 1280x720 尺寸
        fps = 1.0 / (time.time() - t_start + 1e-6)
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Keypoint Tracking (1280x720)", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    val_video("/home/wangzhe/ICRA2025/MY/video/Part2Demo.mp4")
