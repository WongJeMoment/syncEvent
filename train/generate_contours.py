import os
import cv2
import numpy as np

# ==== 配置 ====
input_path = '/home/wangzhe/ICRA2025/MY/data/223.jpg'     # 修改为你的图片路径
output_path = '/home/wangzhe/ICRA2025/MY/train/example.png'  # 输出路径
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ==== 加载并缩放 ====
img = cv2.imread(input_path)
img = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==== 预处理：高斯模糊去燥 + 自适应直方图均衡 ====
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(blurred)

# ==== 自动阈值 Canny 边缘检测 ====
med = np.median(enhanced)
lower = int(max(0, 0.66 * med))
upper = int(min(255, 1.33 * med))
edges = cv2.Canny(enhanced, lower, upper)

# ==== 去噪：形态学闭运算填补小断点 ====
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# ==== 去小噪点：连通域筛选（面积小于阈值的全部去掉） ====
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
min_area = 30  # 面积小于这个值的全部干掉
filtered = np.zeros_like(closed)
for i in range(1, num_labels):  # 跳过背景
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        filtered[labels == i] = 255

# ==== 保存 ====
cv2.imwrite(output_path, filtered)
print(f"✅ 已保存清理后的轮廓图：{output_path}")