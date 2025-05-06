import numpy as np
from cv2 import solvePnP, SOLVEPNP_EPNP
from projection_utils import project_points, compute_confidence

def pecp_pose_estimation(pts_3d, pts_2d, contour_map, contour_pts_3d, camera_matrix,
                         num_iter=500, confidence_thresh=0.33):
    N = pts_3d.shape[0]
    confidences = np.zeros(N)
    m = contour_pts_3d.shape[0]
    threshold = confidence_thresh * m

    for _ in range(num_iter):
        idx = np.random.choice(N, 4, replace=False)
        obj_pts = pts_3d[idx]
        img_pts = pts_2d[idx]

        success, rvec, tvec = solvePnP(obj_pts, img_pts, camera_matrix, None, flags=SOLVEPNP_EPNP)
        if not success:
            continue

        proj_pts = project_points(contour_pts_3d, rvec, tvec, camera_matrix)
        conf = compute_confidence(contour_map, proj_pts)

        if conf > threshold:
            confidences[idx] += (conf - threshold)

    top_idxs = np.argsort(confidences)[-4:]
    best_pts_3d = pts_3d[top_idxs]
    best_pts_2d = pts_2d[top_idxs]

    _, best_rvec, best_tvec = solvePnP(best_pts_3d, best_pts_2d, camera_matrix, None, flags=SOLVEPNP_EPNP)
    return best_rvec, best_tvec
