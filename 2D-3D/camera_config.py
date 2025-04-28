import numpy as np

def get_camera_matrix(image_width, image_height):
    """
    给定图像尺寸，生成相机内参矩阵
    """
    fx = fy = 1200  # 可以根据需要调整
    cx = image_width / 2
    cy = image_height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return camera_matrix
