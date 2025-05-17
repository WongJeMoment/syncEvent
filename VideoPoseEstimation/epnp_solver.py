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


def solve_pnp_epnp(object_points, image_points, camera_matrix,
                   use_ransac=True, reproj_threshold=3.0, ransac_iter=100):
    """
    使用 EPnP 算法求解相机位姿，可选 RANSAC 版本。

    参数:
        object_points: (N, 3) ndarray, 3D 模型点
        image_points : (N, 2) ndarray, 2D 图像点
        camera_matrix: (3, 3) ndarray, 相机内参
        use_ransac   : 是否启用 RANSAC（默认开启）
        reproj_threshold: 内点重投影阈值（单位：像素）
        ransac_iter  : 最大迭代次数

    返回:
        rvec: (3, 1) 旋转向量
        tvec: (3, 1) 平移向量
    """
    object_points = np.array(object_points, dtype=np.float32).reshape(-1, 1, 3)
    image_points  = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    dist_coeffs   = np.zeros((5, 1), dtype=np.float32)  # 默认无畸变

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=reproj_threshold,
            confidence=0.99,
            iterationsCount=ransac_iter
        )
    else:
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
