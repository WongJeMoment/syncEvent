import cv2
import numpy as np

def compute_gradients(surface):
    """
    计算时间表面的空间梯度（X方向和Y方向）。

    参数:
        surface (np.ndarray): 输入时间表面（正极性或负极性），二维数组

    返回:
        grad_x (np.ndarray): X方向的梯度
        grad_y (np.ndarray): Y方向的梯度
    """
    surface_f = surface.astype(np.float32)

    # 计算 X 和 Y 方向梯度，使用 Sobel 算子
    grad_x = cv2.Sobel(surface_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(surface_f, cv2.CV_32F, 0, 1, ksize=3)

    return grad_x, grad_y
