import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from cv2 import solvePnP, SOLVEPNP_EPNP

def project_points(points_3d, rvec, tvec, camera_matrix):
    """将3D点投影到2D图像平面"""
    proj_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
    return proj_points.reshape(-1, 2)

def compute_confidence(contour_map, proj_points, sigma=3):
    """用高斯卷积后的轮廓图评分姿态"""
    heatmap = gaussian_filter(contour_map.astype(np.float32), sigma=sigma)
    h, w = heatmap.shape
    confidence = 0
    for x, y in proj_points:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:
            confidence += heatmap[y, x]
    return confidence

def pecp_pose_estimation(pts_3d, pts_2d, contour_map, contour_pts_3d, camera_matrix,
                         num_iter=500, confidence_thresh=0.33):
    """
    使用PECP进行姿态估计
    - pts_3d: N x 3 已知关键点
    - pts_2d: N x 2 预测关键点
    - contour_map: H x W 的预测轮廓图像（二值图）
    - contour_pts_3d: M x 3 模型轮廓点集
    - camera_matrix: 内参矩阵
    """
    N = pts_3d.shape[0]
    confidences = np.zeros(N)
    m = contour_pts_3d.shape[0]
    threshold = confidence_thresh * m

    for _ in range(num_iter):
        idx = np.random.choice(N, 4, replace=False)
        obj_pts = pts_3d[idx]
        img_pts = pts_2d[idx]

        success, rvec, tvec = solvePnP(obj_pts, img_pts, camera_matrix, None, flags=SOLVEPNP_EPNP)
        if not success:
            continue

        proj_pts = project_points(contour_pts_3d, rvec, tvec, camera_matrix)
        conf = compute_confidence(contour_map, proj_pts)

        if conf > threshold:
            confidences[idx] += (conf - threshold)

    # 按置信度排序，选择最高的四个关键点
    top_idxs = np.argsort(confidences)[-4:]
    best_pts_3d = pts_3d[top_idxs]
    best_pts_2d = pts_2d[top_idxs]

    # 最终姿态估计
    _, best_rvec, best_tvec = solvePnP(best_pts_3d, best_pts_2d, camera_matrix, None, flags=SOLVEPNP_EPNP)
    return best_rvec, best_tvec
