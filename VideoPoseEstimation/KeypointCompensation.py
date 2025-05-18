import cv2
import numpy as np


def project_model_keypoints_to_image(model_points, rvec, tvec, camera_matrix):
    """
    将 3D 模型关键点投影到图像平面，用于跟踪修正。

    参数:
        model_points: (N, 3) np.ndarray，模型空间的关键点
        rvec: (3, 1) 旋转向量
        tvec: (3, 1) 平移向量
        camera_matrix: (3, 3) 相机内参

    返回:
        projected_points: (N, 2) 图像平面坐标
    """
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 默认无畸变

    model_points = np.array(model_points, dtype=np.float32).reshape(-1, 1, 3)

    projected, _ = cv2.projectPoints(
        model_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return projected.reshape(-1, 2)
