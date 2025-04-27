# pnp_solver.py

import os
import numpy as np
import cv2
import torch
from models.model import HeatmapUNet  # 用你的model.py
from models.config import *  # 用你的config.py
from models.dataset import preprocess_image  # 用你的dataset.py
from models.val_video import extract_peak_coords  # 直接复用你的角点提取函数


# --------- solvePnP姿态求解函数 ---------
def solve_pnp(object_points, image_points, camera_matrix, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("❌ solvePnP失败，请检查输入点！")

    return rvec, tvec

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
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)  # 黄色圆圈
        cv2.putText(img, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 255), 2)  # 紫色编号

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

    # 读一张图片（你可以替换路径）
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

    image_points = np.array(keypoints, dtype=np.float32)

    # 你的立方体3D角点坐标 (单位立方体)
    object_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, -1],
        [1, 1, -1],
        [0, 1, -1]
    ], dtype=np.float32)

    # 相机内参
    fx = fy = 800  # 你可以换成真实标定值
    cx = orig_w / 2
    cy = orig_h / 2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 求PnP
    try:
        rvec, tvec = solve_pnp(object_points, image_points, camera_matrix)
        imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)

        # 在原图上画出立方体
        frame_with_cube = draw_cube_with_keypoints(frame.copy(), imgpts, keypoints)

        # 显示出来
        cv2.imshow("Pose Visualization", frame_with_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("✅ 姿态估计成功！")
        print("旋转向量 rvec：\n", rvec)
        print("平移向量 tvec：\n", tvec)
    except ValueError as e:
        print(str(e))



if __name__ == "__main__":
    main()
