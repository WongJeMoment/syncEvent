import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 你的图像路径
img_path = "/home/wangzhe/ICRA2025/MY/data/part2_train_frame/1.jpg"

# 模型文件路径（你需要提前下载 model.yml.gz 到这个位置）
model_path = "/home/wangzhe/ICRA2025/MY/GradientExteaction/model.yml.gz"

# 检查模型文件是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到：{model_path}")

# 加载 Structured Edge 检测器
edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)

# 读取图像并转为 RGB（StructuredEdgeDetection 要求输入为 RGB）
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"图像未找到：{img_path}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 执行边缘检测（注意输入需要归一化到 0~1）
edges = edge_detector.detectEdges(np.float32(img_rgb) / 255.0)

# 显示边缘图
plt.imshow(edges, cmap='gray')
plt.title('Structured Edge Detection')
plt.axis('off')
plt.show()

# 保存结果
save_path = "/home/wangzhe/ICRA2025/MY/data/edge_results/structured_edge.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, (edges * 255).astype(np.uint8))
print(f"边缘图像已保存至：{save_path}")
