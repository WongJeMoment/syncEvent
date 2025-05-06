# config.py

# ==== 相机内参 ====
FX = 800
FY = 800
CX = 320
CY = 240

CAMERA_MATRIX = [
    [FX, 0,  CX],
    [0,  FY, CY],
    [0,  0,  1]
]

# ==== 文件路径 ====
STL_PATH = "models/object.stl"
CONTOUR_IMAGE_PATH = "data/contour_map.png"
KEYPOINTS_3D_PATH = "data/pts_3d.npy"
KEYPOINTS_2D_PATH = "data/pts_2d.npy"

# ==== PECP参数 ====
NUM_ITER = 500
CONFIDENCE_THRESHOLD = 0.33
