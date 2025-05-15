import os
import time
import torch
import numpy as np
import cv2
import scipy.ndimage
from model import HybridHeatmapUNet
from config import *
from stl import mesh
import trimesh
import json

from epnp_solver import solve_pnp_epnp
from camera_config import get_camera_matrix
from keypoint_map import IMAGE_TO_STL_ID, EPnP_INDEXES


def render_projected_stl_with_normal_zbuffer(img, stl_path, rvec, tvec, camera_matrix):
    your_mesh = mesh.Mesh.from_file(stl_path)
    rendered = img.copy()
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    triangles = []
    for i, tri in enumerate(your_mesh.vectors):
        center = tri.mean(axis=0)
        cam_space = R @ center + t
        depth = cam_space[2]
        triangles.append((depth, i, tri))
    triangles.sort(reverse=True)
    for _, i, triangle in triangles:
        imgpts, _ = cv2.projectPoints(triangle, rvec, tvec, camera_matrix, None)
        pts = np.int32(imgpts).reshape(-1, 2)
        normal = your_mesh.normals[i]
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm
        color = (
            int((normal[0] * 0.5 + 0.5) * 255),
            int((normal[1] * 0.5 + 0.5) * 255),
            int((normal[2] * 0.5 + 0.5) * 255)
        )
        cv2.fillConvexPoly(rendered, pts, color)
    blended = cv2.addWeighted(img, 0.3, rendered, 0.7, 0)
    return blended


def draw_projected_stl_edges(img, stl_path, rvec, tvec, camera_matrix, color=(0, 0, 0)):
    mesh_obj = trimesh.load(stl_path, process=True)
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = trimesh.util.concatenate([g for g in mesh_obj.geometry.values()])
    all_edges = mesh_obj.edges_sorted
    edges_sorted, edges_count = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = edges_sorted[edges_count == 1]
    vertices = mesh_obj.vertices
    imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, None)
    imgpts = imgpts.squeeze().astype(int)
    for edge in boundary_edges:
        pt1 = tuple(imgpts[edge[0]])
        pt2 = tuple(imgpts[edge[1]])
        cv2.line(img, pt1, pt2, (0, 150, 150), 20)
    return img


def load_model_points_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    ids = [d["id"] for d in data]
    points = np.array([[d["x"], d["y"], d["z"]] for d in data], dtype=np.float32)
    return points, ids


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


def draw_keypoints_only(img, keypoints):
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(img, str(i), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 1)
    return img


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

    # 视频保存
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
            if len(keypoints) < 4:
                print(f"❌ 检测到关键点数量过少，当前是 {len(keypoints)}，PnP 失败")
                return

            image_points = np.array([keypoints[i] for i in EPnP_INDEXES], dtype=np.float32)
            selected_object_ids = [IMAGE_TO_STL_ID[i] for i in EPnP_INDEXES]
            selected_object_points = np.array(
                [object_points[object_ids.index(i)] for i in selected_object_ids],
                dtype=np.float32
            )
            rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

            frame_with_kps = draw_keypoints_only(frame_resized.copy(), keypoints)
            frame_with_stl = render_projected_stl_with_normal_zbuffer(frame_with_kps, stl_path, rvec, tvec, camera_matrix)
            frame_with_edges = draw_projected_stl_edges(frame_with_stl, stl_path, rvec, tvec, camera_matrix)

            prev_pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
            prev_gray = gray
        else:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            prev_gray = gray.copy()
            prev_pts = next_pts
            tracked_keypoints = next_pts.reshape(-1, 2)

            if len(tracked_keypoints) >= max(EPnP_INDEXES) + 1:
                image_points = np.array([tracked_keypoints[i] for i in EPnP_INDEXES], dtype=np.float32)
                selected_object_ids = [IMAGE_TO_STL_ID[i] for i in EPnP_INDEXES]
                selected_object_points = np.array(
                    [object_points[object_ids.index(i)] for i in selected_object_ids],
                    dtype=np.float32
                )
                rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)
                frame_with_kps = draw_keypoints_only(frame_resized.copy(), tracked_keypoints)
                frame_with_stl = render_projected_stl_with_normal_zbuffer(frame_with_kps, stl_path, rvec, tvec, camera_matrix)
                frame_with_edges = draw_projected_stl_edges(frame_with_stl, stl_path, rvec, tvec, camera_matrix)
            else:
                print(f"❌ 后续帧关键点数量不足，当前为 {len(tracked_keypoints)}，跳过PnP解算")
                frame_with_edges = frame_resized.copy()

        # 帧率显示
        fps = 1.0 / (time.time() - t_start + 1e-6)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_avg = sum(fps_history) / len(fps_history)

        cv2.putText(frame_with_edges, f"FPS: {fps_avg:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame_with_edges, f"Frame: {frame_id}/{total_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 显示与写入视频
        cv2.imshow("Keypoint Tracking (1280x720)", frame_with_edges)
        out.write(frame_with_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    val_video("/home/wangzhe/ICRA2025/MY/video/Part2Demo.mp4")
