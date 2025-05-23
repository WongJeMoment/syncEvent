import numpy as np
import cv2

def is_occluded(pt, gray, border=5, edge_thresh=50):
    """基于 Sobel 的遮挡判断"""
    x, y = int(pt[0]), int(pt[1])
    h, w = gray.shape
    if x < border or y < border or x >= w-border or y >= h-border:
        return True

    patch = gray[y-3:y+4, x-3:x+4]
    if patch.shape != (7, 7):
        return True

    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobelx**2 + sobely**2).mean()

    return edge_strength < edge_thresh

