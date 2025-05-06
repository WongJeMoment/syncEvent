import cv2
from projection_utils import project_points

def visualize_pose(image, rvec, tvec, camera_matrix, model_points, color=(0, 255, 0)):
    proj_points = project_points(model_points, rvec, tvec, camera_matrix)
    proj_points = proj_points.astype(int)
    for (x, y) in proj_points:
        cv2.circle(image, (x, y), 2, color, -1)
    return image
