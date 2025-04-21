import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

IMG_SIZE = 256
HEATMAP_SIZE = 256
SIGMA = 7

image_folder = '/home/wangzhe/ICRA2025/MY/train/frame'  # 输入图像文件夹
json_folder = '/home/wangzhe/ICRA2025/MY/train/json'  # 每张图一个 json
output_folder = '/home/wangzhe/ICRA2025/MY/train/heatmaps'  # 输出 .npy 热图
os.makedirs(output_folder, exist_ok=True)


def generate_heatmap(keypoints, img_size, sigma):
    heatmaps = np.zeros((len(keypoints), img_size, img_size), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0 or x >= img_size or y >= img_size:
            continue
        xx, yy = np.meshgrid(np.arange(img_size), np.arange(img_size))
        heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmaps


# 遍历图像文件夹
for fname in os.listdir(image_folder):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, fname)
    json_path = os.path.join(json_folder, os.path.splitext(fname)[0] + '.json')

    if not os.path.exists(json_path):
        print(f"❌ No matching JSON for {fname}, skipping.")
        continue

    # 加载图像和 JSON 关键点
    img_original = cv2.imread(img_path)
    h, w = img_original.shape[:2]
    img = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))

    with open(json_path, 'r') as f:
        keypoints = json.load(f)

    # 缩放关键点
    scale_x = IMG_SIZE / w
    scale_y = IMG_SIZE / h
    kps = np.array(keypoints, dtype=np.float32)
    kps[:, 0] *= scale_x
    kps[:, 1] *= scale_y

    # 生成热图
    heatmap = generate_heatmap(kps, HEATMAP_SIZE, SIGMA)

    # 保存热图
    output_path = os.path.join(output_folder, os.path.splitext(fname)[0] + '.npy')
    np.save(output_path, heatmap)
    print(f"✅ Heatmap saved: {output_path}")

    # （可选）可视化
    combined = heatmap.sum(axis=0)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(combined, cmap='jet', alpha=0.4)
    plt.title(f"Overlayed Heatmap: {fname}")
    plt.axis('off')
    plt.show()
