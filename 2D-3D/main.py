import os
import cv2
import torch
import numpy as np

from models.model import HeatmapUNet
from models.config import *
from models.dataset import preprocess_image
from models.val_video import extract_peak_coords

from epnp_solver import select_epnp_four_points, solve_pnp_epnp
from camera_config import get_camera_matrix
from object_model import get_cube_model_points
from visualization import draw_cube_with_keypoints

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = HeatmapUNet(num_keypoints=8).to(device)
    best_model_path = "/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt"
    assert os.path.exists(best_model_path), "没有找到最佳模型，请先训练。"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 读图
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

    image_points = select_epnp_four_points(keypoints)
    object_points = get_cube_model_points()

    selected_object_points = np.array([
        object_points[4],  # 点6 (右下后)
        object_points[0],  # 点0 (左上前)
        object_points[1],  # 点1 (右上前)
        object_points[2],  # 点2 (右下前)
    ], dtype=np.float32)

    camera_matrix = get_camera_matrix(orig_w, orig_h)

    try:
        rvec, tvec = solve_pnp_epnp(selected_object_points, image_points, camera_matrix)

        imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)

        frame_with_cube = draw_cube_with_keypoints(frame.copy(), imgpts, keypoints)

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
