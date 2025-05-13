# core/utils.py
import numpy as np
import cv2

def triangulate_points(pt_left, pt_right, cam_matrix_L, cam_matrix_R):
    """
    利用左右图像上的对应点进行三角化，返回三维坐标
    pt_left, pt_right: (u, v)
    cam_matrix_*: 包含 P = K [R|t] 的投影矩阵
    """
    # 默认相机矩阵格式为 P = K @ [R|t]
    pt_left = np.array(pt_left).reshape(2, 1)
    pt_right = np.array(pt_right).reshape(2, 1)

    pts_4d = cv2.triangulatePoints(cam_matrix_L, cam_matrix_R, pt_left, pt_right)
    pts_3d = pts_4d[:3] / pts_4d[3]
    return pts_3d.flatten()

def get_default_camera_matrices():
    """返回默认左右相机的投影矩阵 P = K [R|t]"""
    K = np.array([[240, 0, 128], [0, 240, 128], [0, 0, 1]])
    R = np.eye(3)
    t_L = np.zeros((3, 1))
    t_R = np.array([[0.1, 0, 0]]).T  # baseline = 10cm
    P_L = K @ np.hstack((R, t_L))
    P_R = K @ np.hstack((R, t_R))
    return P_L, P_R