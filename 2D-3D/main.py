import os
import cv2
import torch
import numpy as np
import json
import trimesh
from stl import mesh

from models.model import HybridHeatmapUNet
from models.config import *
from models.val import extract_peak_coords
from epnp_solver import solve_pnp_epnp
from camera_config import get_camera_matrix
from keypoint_map import IMAGE_TO_STL_ID, EPnP_INDEXES


def render_projected_stl_with_normal_zbuffer(img, stl_path, rvec, tvec, camera_matrix):
    """
    带遮挡判断的法线着色渲染（按深度排序）
    """
    your_mesh = mesh.Mesh.from_file(stl_path)
    rendered = img.copy()

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    triangles = []
    for i, tri in enumerate(your_mesh.vectors):
        # 三角形中心点转换到相机坐标系
        center = tri.mean(axis=0)
        cam_space = R @ center + t
        depth = cam_space[2]  # 相机坐标系下 z 越小越近
        triangles.append((depth, i, tri))

    # 按 depth 从远到近排序
    triangles.sort(reverse=True)

    for _, i, triangle in triangles:
        imgpts, _ = cv2.projectPoints(triangle, rvec, tvec, camera_matrix, None)
        pts = np.int32(imgpts).reshape(-1, 2)

        # 法线可视化色彩
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
    """
    画出 STL 模型的边界边（只属于一个面的边），兼容无 edges_unique_faces 的情况
    """
    mesh_obj = trimesh.load(stl_path, process=True)
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = trimesh.util.concatenate([g for g in mesh_obj.geometry.values()])

    # 提取所有面构成的边
    all_edges = mesh_obj.edges_sorted
    edges_sorted, edges_count = np.unique(all_edges, axis=0, return_counts=True)

    # 只属于一个面的边（边界边）
    boundary_edges = edges_sorted[edges_count == 1]

    # 投影所有顶点
    vertices = mesh_obj.vertices
    imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, None)
    imgpts = imgpts.squeeze().astype(int)

    # 画出边界边
    for edge in boundary_edges:
        pt1 = tuple(imgpts[edge[0]])
        pt2 = tuple(imgpts[edge[1]])
        cv2.line(img, pt1, pt2, (0, 150, 150), 20)  # 深蓝色 + 粗一点

    return img


def draw_keypoints_only(img, keypoints):
    """
    只画关键点（不画线）
    """
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)  # 黄色点
        cv2.putText(img, str(i), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 1)  # 紫色编号
    return img


def load_model_points_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    ids = [d["id"] for d in data]
    points = np.array([[d["x"], d["y"], d["z"]] for d in data], dtype=np.float32)
    return points, ids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridHeatmapUNet(num_keypoints=15).to(device)
    best_model_path = "/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt"
    assert os.path.exists(best_model_path), "没有找到最佳模型，请先训练。"
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    img_path = "/home/wangzhe/ICRA2025/MY/data/part2_val_frame/2.jpg"
    assert os.path.exists(img_path), "图片文件不存在！"
    frame = cv2.imread(img_path)
    orig_h, orig_w = frame.shape[:2]

    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmap = model(img_tensor)
    keypoints = extract_peak_coords(pred_heatmap, scale_x=1.0, scale_y=1.0)

    if len(keypoints) < 4:
        print(f"❌ 检测到关键点数量过少，当前是 {len(keypoints)}，PnP 失败")
        return

    model_json_path = "/home/wangzhe/ICRA2025/MY/STL/PART2/Part2.json"
    object_points, object_ids = load_model_points_from_json(model_json_path)

    image_points = np.array([keypoints[i] for i in EPnP_INDEXES], dtype=np.float32)
    selected_object_ids = [IMAGE_TO_STL_ID[i] for i in EPnP_INDEXES]
    selected_object_points = np.array(
        [object_points[object_ids.index(i)] for i in selected_object_ids],
        dtype=np.float32
    )

    camera_matrix = get_camera_matrix(orig_w, orig_h)
    rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

    # ✅ 渲染流程
    stl_path = "/home/wangzhe/ICRA2025/MY/STL/PART2/Part2.STL"
    frame_with_kps = draw_keypoints_only(frame.copy(), keypoints)
    frame_with_stl = render_projected_stl_with_normal_zbuffer(frame_with_kps, stl_path, rvec, tvec, camera_matrix)
    frame_with_edges = draw_projected_stl_edges(frame_with_stl, stl_path, rvec, tvec, camera_matrix)

    # ✅ 显示
    cv2.imshow("STL Render + Edges + Keypoints", frame_with_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ 姿态估计成功！（EPnP版）")
    print("旋转向量 rvec：\n", rvec)
    print("平移向量 tvec：\n", tvec)


if __name__ == "__main__":
    main()
