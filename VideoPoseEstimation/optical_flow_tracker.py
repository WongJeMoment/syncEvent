# optical_flow_tracker.py

import cv2
import numpy as np

def track_keypoints(prev_gray, curr_gray, prev_pts, fb_thresh=1.5):
    """
    持续关键点追踪器（不会让点消失）：
    - 保持关键点顺序不变
    - 跟踪失败时保留上一帧位置
    - 允许后续帧继续追踪这些点

    参数:
        prev_gray: 前一帧灰度图
        curr_gray: 当前帧灰度图
        prev_pts: 前一帧关键点 Nx1x2 float32

    返回:
        next_pts_fixed: 当前帧关键点 Nx1x2（全保留）
        status: Nx1（1=成功，0=失败，但仍保留位置）
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    prev_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, next_pts, None, **lk_params)

    fb_err = np.linalg.norm(prev_pts - prev_back, axis=2).reshape(-1)
    status = (status_fwd.reshape(-1) == 1) & (status_bwd.reshape(-1) == 1) & (fb_err < fb_thresh)

    # 默认：所有关键点都保留，失败点用 prev_pts 原位置补上
    next_pts_fixed = next_pts.copy()
    for i, ok in enumerate(status):
        if not ok:
            next_pts_fixed[i] = prev_pts[i]  # 保留原始位置

    return next_pts_fixed, status.reshape(-1, 1).astype(np.uint8)
