import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def make_odd(k):
    """确保高斯核尺寸为奇数"""
    return k if k % 2 == 1 else k + 1


def gradient_extraction(I, B=3):
    """
    实现梯度提取（Difference-of-Gaussians 方法）

    参数：
        I: 输入图像 (BGR or 灰度)
        B: 高斯模糊基准核大小（推荐为3或5）

    返回：
        E: 最终 DoG 风格梯度图
    """
    # 转灰度图
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) if len(I.shape) == 3 else I

    # Sobel 梯度提取
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 高斯模糊处理
    ksize1 = make_odd(B)
    ksize2 = make_odd(2 * B)
    blur1 = cv2.GaussianBlur(grad_mag, (ksize1, ksize1), 0)
    blur2 = cv2.GaussianBlur(grad_mag, (ksize2, ksize2), 0)

    # DoG
    E = blur1 - blur2

    return E


if __name__ == "__main__":
    # 读取图像路径
    img_path = '/home/wangzhe/ICRA2025/MY/data/part2_train_frame/1.jpg'
    save_path = '/home/wangzhe/ICRA2025/MY/data/output_dog.jpg'

    # 加载图像
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"图像未找到：{img_path}")

    # 提取梯度
    E = gradient_extraction(img, B=5)

    # 显示图像
    plt.imshow(E, cmap='gray')
    plt.title('Extracted Gradient (DoG)')
    plt.axis('off')
    plt.show()

    # 保存结果（归一化后保存）
    E_norm = cv2.normalize(E, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(save_path, E_norm.astype(np.uint8))
    print(f"结果保存至：{save_path}")
