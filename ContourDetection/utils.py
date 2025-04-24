import os
import cv2
import numpy as np
from config import IMAGE_DIR, EDGE_DIR


def generate_canny_edges():
    os.makedirs(EDGE_DIR, exist_ok=True)

    # 遍历图像文件夹中的所有图像
    for fname in os.listdir(IMAGE_DIR):
        if fname.endswith((".jpg", ".png")):
            img_path = os.path.join(IMAGE_DIR, fname)

            try:
                img = cv2.imread(img_path, 0)  # 读取为灰度图
                if img is None:
                    print(f"错误：无法读取图像 {fname}，跳过该文件。")
                    continue

                # 对比度增强（使用CLAHE）
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)

                # 高斯滤波，去除噪声并保留边缘
                img_blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

                # 动态计算 Canny 阈值
                median = np.median(img_blurred)
                lower_threshold = int(max(0, 0.7 * median))  # 增加低阈值的灵敏度
                upper_threshold = int(min(255, 2 * median))  # 增加高阈值的灵敏度

                # Canny 边缘检测
                edge = cv2.Canny(img_blurred, lower_threshold, upper_threshold)

                # 保存边缘图
                edge_path = os.path.join(EDGE_DIR, fname)
                cv2.imwrite(edge_path, edge)

                print(f"已处理: {fname}")
            except Exception as e:
                print(f"处理图像 {fname} 时出错: {e}")


def main():
    print("开始边缘检测过程...")
    generate_canny_edges()


if __name__ == "__main__":
    main()
