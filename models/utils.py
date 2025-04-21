# utils.py
import numpy as np
import cv2

def generate_heatmap(keypoints, img_size, sigma):
    heatmaps = np.zeros((len(keypoints), img_size, img_size), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0 or x >= img_size or y >= img_size:
            continue
        xx, yy = np.meshgrid(np.arange(img_size), np.arange(img_size))
        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmaps
