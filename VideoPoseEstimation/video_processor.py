import cv2
import numpy as np
import torch
import os
import time

from keypoint_utils import preprocess_image, pad_to_multiple, extract_peak_coords, draw_keypoints_only
from stl_renderer import render_projected_stl_with_normal_zbuffer, draw_projected_stl_edges
from pose_utils import load_model_points_from_json
from epnp_solver import solve_pnp_epnp
from camera_config import get_camera_matrix
from model import HybridHeatmapUNet
from keypoint_map import IMAGE_TO_STL_ID, EPnP_INDEXES
from optical_flow_tracker import track_keypoints
from DrawSTLPoints import draw_projected_keypoints


def selective_correct_keypoints(tracked_kps, stl_object_points, rvec, tvec, camera_matrix, mapping_dict,
                                threshold=10.0):
    """
    选择性地修正偏移过大的关键点：
    - 若 tracked_kps 与其对应 STL 投影点偏移大于 threshold，则重置为 STL 投影点；
    - 否则保持跟踪值。

    参数：
        tracked_kps: np.ndarray, shape=(N, 2)，光流跟踪的关键点
        stl_object_points: list of 3D 点（STL中所有关键点）
        rvec, tvec: 当前帧的位姿估计
        camera_matrix: 内参矩阵
        mapping_dict: dict，图像关键点索引 → STL 点索引
        threshold: float，偏移阈值（单位：像素）

    返回：
        corrected_kps: np.ndarray, shape=(N, 2)，已修正的关键点
    """
    corrected = tracked_kps.copy()
    stl_pts = np.array(stl_object_points, dtype=np.float32)

    # 投影 STL 点到图像平面
    projected_pts, _ = cv2.projectPoints(stl_pts, rvec, tvec, camera_matrix, distCoeffs=None)
    projected_pts = projected_pts.reshape(-1, 2)

    for img_idx, stl_idx in mapping_dict.items():
        if img_idx < len(tracked_kps) and stl_idx < len(projected_pts):
            dist = np.linalg.norm(tracked_kps[img_idx] - projected_pts[stl_idx])
            if dist > threshold:
                corrected[img_idx] = projected_pts[stl_idx]
    return corrected


def smooth_correct_keypoints(tracked_kps, stl_object_points, rvec, tvec, camera_matrix, mapping_dict, alpha=0.2):
    """
    平滑修正跟踪关键点（将其拉向对应 STL 投影点）
    """
    corrected = tracked_kps.copy()
    stl_pts = np.array(stl_object_points, dtype=np.float32)

    # 投影所有 STL 点
    projected_pts, _ = cv2.projectPoints(stl_pts, rvec, tvec, camera_matrix, distCoeffs=None)
    projected_pts = projected_pts.reshape(-1, 2)

    for img_idx, stl_idx in mapping_dict.items():
        if img_idx < len(tracked_kps) and stl_idx < len(projected_pts):
            corrected[img_idx] = (1 - alpha) * tracked_kps[img_idx] + alpha * projected_pts[stl_idx]
    return corrected

def val_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridHeatmapUNet(num_keypoints=15).to(device)
    model_path = "/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt"
    assert os.path.exists(model_path), "❌ 未找到模型"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    orig_h, orig_w = 1280, 720

    model_json_path = "/home/wangzhe/ICRA2025/MY/STL/PART2/Part2.json"
    object_points, object_ids = load_model_points_from_json(model_json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return

    save_path = "output_tracking.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (1280, 720))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    prev_pts = None
    frame_id = 0
    fps_history = []

    stl_path = "/home/wangzhe/ICRA2025/MY/STL/PART2/Part2.STL"
    camera_matrix = get_camera_matrix(orig_w, orig_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ 视频结束")
            break

        t_start = time.time()
        frame_resized = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_id += 1

        if prev_pts is None:
            padded_img, orig_hw = pad_to_multiple(frame_resized)
            img_input = preprocess_image(padded_img)
            img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_heatmap = model(img_tensor).cpu()
                pred_heatmap = pred_heatmap[:, :, :orig_hw[0], :orig_hw[1]]

            keypoints = extract_peak_coords(pred_heatmap, orig_size=orig_hw)
            print(f"📌 第1帧检测关键点数: {len(keypoints)}")
            if len(keypoints) < max(EPnP_INDEXES) + 1:
                print(f"❌ 关键点数量不足，跳过（{len(keypoints)} < {max(EPnP_INDEXES)+1}）")
                return

            image_points = np.array([keypoints[i] for i in EPnP_INDEXES], dtype=np.float32)
            selected_object_ids = [IMAGE_TO_STL_ID[i] for i in EPnP_INDEXES]
            selected_object_points = np.array([object_points[object_ids.index(i)] for i in selected_object_ids], dtype=np.float32)

            rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

            frame_with_kps = draw_keypoints_only(frame_resized.copy(), keypoints)
            frame_with_stl = render_projected_stl_with_normal_zbuffer(frame_with_kps, stl_path, rvec, tvec, camera_matrix)
            frame_with_edges = draw_projected_stl_edges(frame_with_stl, stl_path, rvec, tvec, camera_matrix)
            frame_with_edges = draw_projected_keypoints(frame_with_edges, np.array(object_points, dtype=np.float32), rvec, tvec, camera_matrix)

            prev_pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
            prev_gray = gray

        else:
            next_pts, status = track_keypoints(prev_gray, gray, prev_pts)
            prev_gray = gray.copy()
            tracked_keypoints = next_pts.reshape(-1, 2)

            if len(tracked_keypoints) < max(EPnP_INDEXES) + 1:
                print("❌ 跟踪点数量不足，跳过")
                prev_pts = None
                frame_with_edges = frame_resized.copy()
                continue

            image_points = np.array([tracked_keypoints[i] for i in EPnP_INDEXES], dtype=np.float32)
            selected_object_ids = [IMAGE_TO_STL_ID[i] for i in EPnP_INDEXES]
            selected_object_points = np.array([object_points[object_ids.index(i)] for i in selected_object_ids], dtype=np.float32)

            rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

            # 替换为动态偏移阈值 + 选择性重置策略
            tracked_keypoints = selective_correct_keypoints(
                tracked_keypoints,
                object_points,
                rvec,
                tvec,
                camera_matrix,
                IMAGE_TO_STL_ID,
                threshold=10.0  # 可根据实际调节为 5~15
            )

            prev_pts = tracked_keypoints.reshape(-1, 1, 2)

            frame_with_kps = draw_keypoints_only(frame_resized.copy(), tracked_keypoints)
            frame_with_stl = render_projected_stl_with_normal_zbuffer(frame_with_kps, stl_path, rvec, tvec, camera_matrix)
            frame_with_edges = draw_projected_stl_edges(frame_with_stl, stl_path, rvec, tvec, camera_matrix)
            frame_with_edges = draw_projected_keypoints(frame_with_edges, np.array(object_points, dtype=np.float32), rvec, tvec, camera_matrix)

        fps = 1.0 / (time.time() - t_start + 1e-6)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_avg = sum(fps_history) / len(fps_history)

        cv2.putText(frame_with_edges, f"FPS: {fps_avg:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame_with_edges, f"Frame: {frame_id}/{total_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Keypoint Tracking (1280x720)", frame_with_edges)
        out.write(frame_with_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
