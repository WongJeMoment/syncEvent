import numpy as np
import cv2
from pose_estimator import pecp_pose_estimation
from visualization import visualize_pose
from stl_utils import load_stl_contour_points
from config import CAMERA_MATRIX, STL_PATH, CONTOUR_IMAGE_PATH, NUM_ITER, CONFIDENCE_THRESHOLD

def main():
    # 载入关键点
    pts_3d = np.load("data/pts_3d.npy")
    pts_2d = np.load("data/pts_2d.npy")

    # 轮廓图
    contour_map = cv2.imread(CONTOUR_IMAGE_PATH, 0)

    # 载入 STL 轮廓点
    contour_pts_3d = load_stl_contour_points(STL_PATH, sample_num=500)

    # 姿态估计
    rvec, tvec = pecp_pose_estimation(
        pts_3d, pts_2d, contour_map, contour_pts_3d,
        np.array(CAMERA_MATRIX, dtype=np.float32),
        num_iter=NUM_ITER,
        confidence_thresh=CONFIDENCE_THRESHOLD
    )

    print("rvec:", rvec.ravel())
    print("tvec:", tvec.ravel())

    # 可视化
    color_img = cv2.cvtColor(contour_map, cv2.COLOR_GRAY2BGR)
    vis = visualize_pose(color_img, rvec, tvec, np.array(CAMERA_MATRIX, dtype=np.float32), contour_pts_3d)
    cv2.imshow("Projection", vis)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
