import os
import time
import torch
import numpy as np
import cv2
import scipy.ndimage
from model import HybridHeatmapUNet
from config import *

torch.backends.cudnn.benchmark = True

# ---------- 图像预处理 ----------
def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img

def pad_to_multiple(img, divisor=32):
    h, w = img.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return img_padded, (h, w)

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
            if not visited[j] and np.hypot(x_i - x_j, y_i - y_j) < merge_distance:
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

# ---------- 可视化 ----------
def visualize_keypoints(img, keypoints):
    for idx, (x, y) in enumerate(keypoints):
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(img, str(idx), (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return img

# ---------- 主函数 ----------
def val_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridHeatmapUNet(num_keypoints=15).to(device)
    model_path = "/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt"
    assert os.path.exists(model_path), "❌ 未找到模型"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return

    # ---------- 保存视频设置 ----------
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_video / 2  # 放慢一倍
    width_out, height_out = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "output_keypoints_slow.mp4"
    video_writer = cv2.VideoWriter(save_path, fourcc, fps_out, (width_out, height_out))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    prev_pts = None
    frame_id = 0
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ 视频结束")
            break

        t_start = time.time()
        frame_resized = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_id += 1

        if frame_id == 1:
            # ✅ 第一帧使用模型检测
            padded_img, orig_hw = pad_to_multiple(frame_resized)
            img_input = preprocess_image(padded_img)
            img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_heatmap = model(img_tensor).cpu()
                pred_heatmap = pred_heatmap[:, :, :orig_hw[0], :orig_hw[1]]

            keypoints = extract_peak_coords(pred_heatmap, orig_size=orig_hw)
            prev_pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
            prev_gray = gray
        else:
            # ✅ 后续帧使用光流跟踪
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            prev_gray = gray.copy()
            prev_pts = next_pts

        # ✅ 可视化关键点
        vis_frame = frame_resized.copy()
        if prev_pts is not None and len(prev_pts) > 0:
            vis_frame = visualize_keypoints(vis_frame, prev_pts.reshape(-1, 2))

        # ✅ 显示帧率和帧编号
        fps = 1.0 / (time.time() - t_start + 1e-6)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_avg = sum(fps_history) / len(fps_history)

        cv2.putText(vis_frame, f"FPS: {fps_avg:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Frame: {frame_id}/{total_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # ✅ 保存帧到视频
        video_writer.write(vis_frame)

        # ✅ 显示窗口
        cv2.imshow("Keypoint Tracking (Slow Save)", vis_frame)
        if cv2.waitKey(31) & 0xFF == ord('q'):
            break

    # ---------- 释放资源 ----------
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"🎥 视频已保存至：{save_path}")

if __name__ == "__main__":
    val_video("/home/wangzhe/ICRA2025/MY/video/Part2Demo.mp4")
