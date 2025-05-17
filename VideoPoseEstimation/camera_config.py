import numpy as np

def get_camera_matrix(image_width, image_height):
    """
    给定图像尺寸，生成相机内参矩阵
    """
    fx = 1761.63991067404
    fy = 1763.02728468010  # 可以根据需要调整
    cx = 672.771832431268
    cy = 364.906730287696

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return camera_matrix
