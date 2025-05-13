import torch
import numpy as np

def build_kew(events, kp, r=10, tau=20000, N_max=256):
    """
    构建单个关键点的 KEW（关键点锚定事件窗口）
    events: Tensor[N, 4]，(x, y, t, p)
    kp: Tuple (u, v)，关键点位置
    return: Tensor[M, 4]，保留的事件
    """
    x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    u, v = kp
    dist2 = (x - u)**2 + (y - v)**2
    time_mask = t > (t.max() - tau)
    mask = (dist2 < r**2) & time_mask
    selected = events[mask]
    selected = selected[torch.argsort(selected[:, 2], descending=True)]
    return selected[:N_max]

def build_kew_batch(kp_list, left_events, right_events, r=10, tau=20000, N_max=256):
    """
    批量构建左右目 KEW，分别对每个关键点构建一个事件窗口
    返回：两个 list，每个元素是 Tensor[M, 4]
    """
    kew_L = [build_kew(left_events, kp, r, tau, N_max) for kp in kp_list]
    kew_R = [build_kew(right_events, kp, r, tau, N_max) for kp in kp_list]
    return kew_L, kew_R
