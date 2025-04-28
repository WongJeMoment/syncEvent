import numpy as np
import cv2

def select_epnp_four_points(keypoints):
    """
    选出用于EPnP的四个关键点：[6, 0, 1, 2]
    """
    if len(keypoints) != 8:
        raise ValueError(f"需要8个关键点，目前是{len(keypoints)}个！")

    selected_idx = [6, 0, 1, 2]
    selected_points = np.array([keypoints[i] for i in selected_idx], dtype=np.float32)

    return selected_points

def solve_pnp_epnp(object_points, image_points, camera_matrix):
    """
    用EPnP算法求解rvec, tvec
    """
    dist_coeffs = np.zeros((5, 1))  # 默认无畸变

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        raise ValueError("❌ EPnP求解失败，请检查输入点！")

    return rvec, tvec
