import cv2
import numpy as np
from keypoint_tracker.PointsKalman import PointKalman

def track_keypoints(prev_gray, curr_gray, prev_pts, kf_bank=None,
                    threshold=2.0, structure_edges=None, structure_tol=0.2):
    """
    四向光流一致性 + Kalman(6维) + 自适应过程噪声 + 结构保持
    """
    if prev_pts is None or len(prev_pts) == 0:
        return prev_pts, np.zeros((0, 1), dtype=np.uint8)

    if kf_bank is None:
        kf_bank = [PointKalman(pt.ravel()) for pt in prev_pts]

    lk_params = dict(
        winSize=(61, 61),
        maxLevel=4,
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-10,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 0.01)
    )

    # 光流四向一致性
    next_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, next_pts, None, **lk_params)
    next2_pts, st_fwd2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, next_pts, None, **lk_params)
    reprojected_next, st_rev2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, next2_pts, None, **lk_params)

    forward_error = np.linalg.norm(prev_pts.squeeze() - back_pts.squeeze(), axis=1)
    round_trip_error = np.linalg.norm(next_pts.squeeze() - reprojected_next.squeeze(), axis=1)

    status = ((st_fwd.flatten() == 1) &
              (st_back.flatten() == 1) &
              (st_fwd2.flatten() == 1) &
              (st_rev2.flatten() == 1) &
              (forward_error < threshold) &
              (round_trip_error < threshold))

    next_pts_fixed = np.zeros_like(prev_pts)

    for i, ok in enumerate(status):
        pt_kf = kf_bank[i].predict()

        if ok:
            pt_lk = next_pts[i].ravel()
            error = (forward_error[i] + round_trip_error[i]) / 2.0

            # ✅ 动态调整 Kalman 的 Q：误差越大，Q 越大（越不信任预测）
            dynamic_scale = 1.0 + (error / threshold)
            kf_bank[i].set_process_noise_scale(dynamic_scale)

            # ✅ 光流置信度加权更新
            confidence = np.exp(-error / threshold)
            confidence = np.clip(confidence, 0.0, 1.0)

            pt = confidence * pt_lk + (1 - confidence) * pt_kf
            kf_bank[i].correct(pt)
        else:
            pt = pt_kf
            status[i] = 0

        next_pts_fixed[i, 0] = pt

    # ✅ 结构保持
    if structure_edges is not None:
        for i, j in structure_edges:
            if i >= len(prev_pts) or j >= len(prev_pts):
                continue
            d_prev = np.linalg.norm(prev_pts[i].ravel() - prev_pts[j].ravel())
            d_curr = np.linalg.norm(next_pts_fixed[i, 0] - next_pts_fixed[j, 0])
            ratio = abs(d_curr - d_prev) / (d_prev + 1e-6)
            if ratio > structure_tol:
                next_pts_fixed[i, 0] = kf_bank[i].predict()
                next_pts_fixed[j, 0] = kf_bank[j].predict()
                status[i] = 0
                status[j] = 0

    return next_pts_fixed, status.reshape(-1, 1).astype(np.uint8)