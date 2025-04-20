import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

IMG_SIZE = 256
HEATMAP_SIZE = 256
SIGMA = 7

image_path = '/home/wangzhe/ICRA2025/MY/data/222.jpg'
label_json = '/home/wangzhe/ICRA2025/MY/data/222.json'
output_path = '/home/wangzhe/ICRA2025/MY/data/222.npy'

def generate_heatmap(keypoints, img_size, sigma):
    heatmaps = np.zeros((len(keypoints), img_size, img_size), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0 or x >= img_size or y >= img_size:
            continue
        xx, yy = np.meshgrid(np.arange(img_size), np.arange(img_size))
        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmaps

# 加载 keypoints
with open(label_json, 'r') as f:
    keypoints_dict = json.load(f)

fname = os.path.basename(image_path)
if fname not in keypoints_dict:
    print(f"❌ Missing keypoints for {fname}")
    exit()

# 读取图像并记录原始大小
img_original = cv2.imread(image_path)
h, w = img_original.shape[:2]

# Resize 图像
img = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))

# 分别计算缩放比例
scale_x = IMG_SIZE / w
scale_y = IMG_SIZE / h

# 缩放关键点
kps = np.array(keypoints_dict[fname], dtype=np.float32)
kps[:, 0] *= scale_x
kps[:, 1] *= scale_y

# 生成热图
heatmap = generate_heatmap(kps, HEATMAP_SIZE, SIGMA)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, heatmap)
print(f"✅ Heatmap saved to: {output_path}")

# 可视化叠加
combined = heatmap.sum(axis=0)
combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(combined, cmap='jet', alpha=0.4)
plt.title("Overlayed Keypoint Heatmap")
plt.axis('off')
plt.show()
