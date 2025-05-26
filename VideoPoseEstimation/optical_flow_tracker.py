import cv2
import numpy as np
from keypoint_tracker.PointsKalman import PointKalman

def track_keypoints(prev_gray, curr_gray, prev_pts, kf_bank=None, threshold=2.0):
    """
    四向光流一致性校验 + Kalman
    （已移除遮挡检测模块）
    """
    if prev_pts is None or len(prev_pts) == 0:
        return prev_pts, np.zeros((0, 1), dtype=np.uint8)

    if kf_bank is None:
        kf_bank = [PointKalman(pt.ravel()) for pt in prev_pts]

    lk_params = dict(
        winSize=(61, 61),
        maxLevel=4,
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
    )

    # 光流前向 & 反向 & round-trip 检查
    next_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, next_pts, None, **lk_params)
    next2_pts, st_fwd2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, next_pts, None, **lk_params)
    reprojected_next, st_rev2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, next2_pts, None, **lk_params)

    forward_error = np.linalg.norm(prev_pts.squeeze() - back_pts.squeeze(), axis=1)
    round_trip_error = np.linalg.norm(next_pts.squeeze() - reprojected_next.squeeze(), axis=1)

    # 四向一致性状态判断
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
            confidence = np.exp(-error / threshold)  # 越小误差置信度越高
            confidence = np.clip(confidence, 0.0, 1.0)
            pt = confidence * pt_lk + (1 - confidence) * pt_kf
            kf_bank[i].correct(pt)
        else:
            pt = pt_kf
            status[i] = 0

        next_pts_fixed[i, 0] = pt

    return next_pts_fixed, status.reshape(-1, 1).astype(np.uint8)
