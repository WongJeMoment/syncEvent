import os
import cv2
import numpy as np

# ----------- 参数配置 -----------
INPUT_DIR = "/home/wangzhe/ICRA2025/MY/data/train_frame"       # 输入图像文件夹
OUTPUT_DIR = "/home/wangzhe/ICRA2025/MY/data/canny_frame"     # 输出图像文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

GAUSS_KERNEL = (5, 5)
CANNY_LOW, CANNY_HIGH = 100, 200
SOBEL_THRESH = 50
MIN_AREA = 50
OPEN_KERNEL = np.ones((3, 3), np.uint8)

# ----------- 图像处理函数 -----------
def process_image(img_path, save_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️ 无法读取图像: {img_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, GAUSS_KERNEL, 0)

    # Canny（红）
    canny_edges = cv2.Canny(gray_blur, CANNY_LOW, CANNY_HIGH)
    canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=1)

    # Sobel（绿）
    sobel_x = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_blur, cv2.CV_16S, 0, 1, ksize=3)
    sobel_mag = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5,
                                cv2.convertScaleAbs(sobel_y), 0.5, 0)

    _, sobel_bin = cv2.threshold(sobel_mag, SOBEL_THRESH, 255, cv2.THRESH_BINARY)
    sobel_bin = cv2.morphologyEx(sobel_bin, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(sobel_bin, connectivity=8)
    clean_sobel = np.zeros_like(sobel_bin)
    for i in range(1, num):  # 忽略背景
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean_sobel[labels == i] = 255

    vis = np.zeros_like(frame)
    vis[:, :, 2] = canny_edges
    vis[:, :, 1] = clean_sobel
    blended = cv2.addWeighted(frame, 0.8, vis, 1.0, 0)

    cv2.imwrite(save_path, blended)
    print(f"✅ Saved: {save_path}")

# ----------- 主函数 -----------
def main():
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".jpg")]
    for img_file in sorted(images):
        input_path = os.path.join(INPUT_DIR, img_file)
        output_path = os.path.join(OUTPUT_DIR, img_file)
        process_image(input_path, output_path)

if __name__ == "__main__":
    main()
