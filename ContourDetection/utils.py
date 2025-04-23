# utils.py
import os
import cv2
from config import IMAGE_DIR, EDGE_DIR

def generate_canny_edges():
    os.makedirs(EDGE_DIR, exist_ok=True)
    for fname in os.listdir(IMAGE_DIR):
        if fname.endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(IMAGE_DIR, fname), 0)
            edge = cv2.Canny(img, 100, 200)
            cv2.imwrite(os.path.join(EDGE_DIR, fname), edge)
