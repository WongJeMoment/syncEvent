import numpy as np

def get_cube_model_points():
    """
    返回立方体的8个三维顶点
    """
    points = np.array([
        [0, 0, 0],    # 0 左上前
        [1, 0, 0],    # 1 右上前
        [1, 1, 0],    # 2 右下前
        [0, 1, 0],    # 3 左下前
        [0, 0, -1],   # 4 左上后
        [1, 0, -1],   # 5 右上后
        [1, 1, -1],   # 6 右下后
        [0, 1, -1],   # 7 左下后
    ], dtype=np.float32)

    return points
