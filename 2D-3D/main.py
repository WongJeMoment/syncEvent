# pnp_solver_epnp.py

import os
import numpy as np
import cv2
import torch
from models.model import HeatmapUNet  # 你的模型
from models.config import *  # 你的配置
from models.dataset import preprocess_image  # 你的预处理
from models.val_video import extract_peak_coords  # 角点提取

# --------- 选指定的4个关键点 ---------
def select_epnp_four_points(keypoints):
    """
    从8个关键点中选出指定的4个，用来做EPnP。
    按顺序：[6, 0, 1, 2]
    """
    if len(keypoints) != 8:
        raise ValueError(f"需要8个关键点，目前是{len(keypoints)}个！")

    selected_idx = [6, 0, 1, 2]
    selected_points = np.array([keypoints[i] for i in selected_idx], dtype=np.float32)

    return selected_points

# --------- EPnP求解函数 ---------
def solve_pnp_epnp(object_points, image_points, camera_matrix):
    dist_coeffs = np.zeros((5, 1))  # 默认无畸变

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP  # 🔥 选EPnP
    )

    if not success:
        raise ValueError("❌ EPnP求解失败，请检查输入点！")

    return rvec, tvec

# --------- 绘制立方体+关键点 ---------
def draw_cube_with_keypoints(img, imgpts, keypoints):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # 画底面
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)
    # 画立柱
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
    # 画顶面
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)

    # 画关键点
    for idx, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(img, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 255), 2)

    return img

# --------- 主程序 ---------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = HeatmapUNet(num_keypoints=8).to(device)
    best_model_path = "/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt"
    assert os.path.exists(best_model_path), "没有找到最佳模型，请先训练。"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 读一张图片
    img_path = "/home/wangzhe/ICRA2025/MY/data/frame/1.jpg"
    assert os.path.exists(img_path), "图片文件不存在！"

    frame = cv2.imread(img_path)
    orig_h, orig_w = frame.shape[:2]

    img_input = preprocess_image(frame)
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmap = model(img_tensor)

    # 提取关键点
    keypoints = extract_peak_coords(pred_heatmap, orig_size=(orig_h, orig_w))

    if len(keypoints) != 8:
        print(f"⚠️ 检测到的角点数量不足8个，目前是{len(keypoints)}个，PnP可能失败！")
        return

    # 选四个关键点
    image_points = select_epnp_four_points(keypoints)

    # 选对应object_points
    full_object_points = np.array([
        [0, 0, 0],    # 0 左上前
        [1, 0, 0],    # 1 右上前
        [1, 1, 0],    # 2 右下前
        [0, 1, 0],    # 3 左下前
        [0, 0, -1],   # 4 左上后
        [1, 0, -1],   # 5 右上后
        [1, 1, -1],   # 6 右下后
        [0, 1, -1],   # 7 左下后
    ], dtype=np.float32)

    selected_object_points = np.array([
        full_object_points[4],  # 点6 (右下后)
        full_object_points[0],  # 点0 (左上前)
        full_object_points[1],  # 点1 (右上前)
        full_object_points[2],  # 点2 (右下前)
    ], dtype=np.float32)

    # 相机内参
    fx = fy = 800
    cx = orig_w / 2
    cy = orig_h / 2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 用EPnP求解姿态
    try:
        rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

        # 投影所有8个点，用来画立方体
        imgpts, _ = cv2.projectPoints(full_object_points, rvec, tvec, camera_matrix, None)

        # 绘制立方体+关键点
        frame_with_cube = draw_cube_with_keypoints(frame.copy(), imgpts, keypoints)

        # 显示
        cv2.imshow("Pose Visualization (EPnP)", frame_with_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("✅ 姿态估计成功！（EPnP版）")
        print("旋转向量 rvec：\n", rvec)
        print("平移向量 tvec：\n", tvec)

    except ValueError as e:
        print(str(e))

if __name__ == "__main__":
    main()
