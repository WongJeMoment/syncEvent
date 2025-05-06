import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from cv2 import solvePnP, SOLVEPNP_EPNP

def project_points(points_3d, rvec, tvec, camera_matrix):
    """将3D点投影到2D图像平面"""
    # 将 3D 点投影成 2D 图像坐标。
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

def visualize_pose(image, rvec, tvec, camera_matrix, model_points, color=(0, 255, 0)):
    """
    将3D模型点投影到图像上，并在图像中绘制轮廓或点
    """
    proj_points = project_points(model_points, rvec, tvec, camera_matrix)
    proj_points = proj_points.astype(int)

    # 绘制轮廓点
    for (x, y) in proj_points:
        cv2.circle(image, (x, y), 2, color, -1)

    return image


def main():
    # === 模拟输入数据 ===
    # 假设我们有8个关键点
    pts_3d = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.float32)

    # 假设相机坐标下对应的2D投影点（这里通常应是模型预测结果）
    pts_2d = np.array([
        [100, 100],
        [100, 200],
        [200, 100],
        [200, 200],
        [110, 110],
        [110, 210],
        [210, 110],
        [210, 210]
    ], dtype=np.float32)

    # 预测轮廓图（模拟为一个高斯圆）
    contour_map = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(contour_map, (160, 160), 100, 255, thickness=2)

    # 模拟3D模型的轮廓点
    contour_pts_3d = np.array([
        [np.cos(theta), np.sin(theta), 0]
        for theta in np.linspace(0, 2*np.pi, 100)
    ], dtype=np.float32)

    # 相机内参矩阵
    fx = fy = 800
    cx, cy = 320, 240
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    # === 调用 PECP 姿态估计 ===
    rvec, tvec = pecp_pose_estimation(
        pts_3d, pts_2d, contour_map, contour_pts_3d, camera_matrix,
        num_iter=500, confidence_thresh=0.33
    )

    print("Estimated rotation vector (rvec):", rvec.ravel())
    print("Estimated translation vector (tvec):", tvec.ravel())

    # 可视化姿态结果
    color_map = cv2.cvtColor(contour_map, cv2.COLOR_GRAY2BGR)
    vis_img = visualize_pose(color_map.copy(), rvec, tvec, camera_matrix, contour_pts_3d)

    # 显示和保存结果
    cv2.imshow("Projected Contour Overlay", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
