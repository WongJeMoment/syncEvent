import os
import cv2
import torch
import numpy as np
import json

from models.model import HeatmapUNet
from models.config import *
from models.dataset import preprocess_image
from models.val_video import extract_peak_coords

from epnp_solver import select_epnp_four_points, solve_pnp_epnp
from camera_config import get_camera_matrix
from visualization import draw_cube_with_keypoints
from stl import mesh
from keypoint_map import IMAGE_TO_STL_ID, EPnP_INDEXES


def draw_projected_stl_on_image(img, stl_path, rvec, tvec, camera_matrix):
    """
    将 STL 模型根据姿态投影到图像中并绘制
    """
    your_mesh = mesh.Mesh.from_file(stl_path)
    projected_img = img.copy()

    for triangle in your_mesh.vectors:
        # Project each vertex
        imgpts, _ = cv2.projectPoints(triangle, rvec, tvec, camera_matrix, None)
        pts = np.int32(imgpts).reshape(-1, 2)

        # Draw triangle
        cv2.polylines(projected_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return projected_img

def load_model_points_from_json(json_path):
    """
    从 JSON 文件中加载 3D 模型点及其 ID
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    ids = [d["id"] for d in data]
    points = np.array([[d["x"], d["y"], d["z"]] for d in data], dtype=np.float32)
    return points, ids

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = HeatmapUNet(num_keypoints=8).to(device)
    best_model_path = "/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt"
    assert os.path.exists(best_model_path), "没有找到最佳模型，请先训练。"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 读取图像
    img_path = "/home/wangzhe/ICRA2025/MY/data/frame/1.jpg"
    assert os.path.exists(img_path), "图片文件不存在！"

    frame = cv2.imread(img_path)
    orig_h, orig_w = frame.shape[:2]

    img_input = preprocess_image(frame)
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmap = model(img_tensor)

    keypoints = extract_peak_coords(pred_heatmap, orig_size=(orig_h, orig_w))

    if len(keypoints) != 8:
        print(f"⚠️ 检测到的角点数量不足8个，目前是{len(keypoints)}个，PnP可能失败！")
        return

    # 加载模型 3D 角点（含 ID）
    model_json_path = "/STL/cube/cube.json"
    object_points, object_ids = load_model_points_from_json(model_json_path)

    # 选择用于EPnP的图像关键点（4个）
    image_points = np.array([keypoints[i] for i in EPnP_INDEXES], dtype=np.float32)

    selected_object_ids = [IMAGE_TO_STL_ID[i] for i in EPnP_INDEXES]

    try:
        # 根据 ID 映射出4个对应的 3D 点
        selected_object_points = np.array(
            [object_points[object_ids.index(i)] for i in selected_object_ids],
            dtype=np.float32
        )

        # 相机参数
        camera_matrix = get_camera_matrix(orig_w, orig_h)

        # PnP求解
        rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

        # 投影3D点用于画图
        imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)

        # 可视化结果
        # 画出 cube 边框和关键点
        frame_with_cube = draw_cube_with_keypoints(frame.copy(), imgpts, keypoints)

        # 画出姿态变换后的 STL 模型
        stl_path = "/STL/cube/cube.STL"  # 替换为你的 STL 路径
        frame_with_stl = draw_projected_stl_on_image(frame_with_cube, stl_path, rvec, tvec, camera_matrix)

        # 显示最终图像
        cv2.imshow("Pose Visualization with STL", frame_with_stl)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("✅ 姿态估计成功！（EPnP版）")
        print("旋转向量 rvec：\n", rvec)
        print("平移向量 tvec：\n", tvec)

    except ValueError as e:
        print("❌ PnP失败：", str(e))

if __name__ == "__main__":
    main()
