import cv2
import numpy as np


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