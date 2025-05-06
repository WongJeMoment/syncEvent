import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def project_points(points_3d, rvec, tvec, camera_matrix):
    proj_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
    return proj_points.reshape(-1, 2)

def compute_confidence(contour_map, proj_points, sigma=3):
    heatmap = gaussian_filter(contour_map.astype(np.float32), sigma=sigma)
    h, w = heatmap.shape
    confidence = 0
    for x, y in proj_points:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:
            confidence += heatmap[y, x]
    return confidence
