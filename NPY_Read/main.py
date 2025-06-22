import numpy as np
from TimeSurfer import build_time_surfaces
from GradientUtils import compute_gradients
from Visualize import show_time_surfaces


def generate_sample_events():
    # 人工构造一个简单的事件序列
    dtype = np.dtype([('t', 'int64'), ('x', 'uint16'), ('y', 'uint16'), ('p', 'int8')])
    events = np.array([
        (1000, 10, 10, 1),
        (2000, 11, 10, 1),
        (3000, 12, 10, 1),
        (4000, 13, 10, -1),
        (5000, 14, 10, -1),
        (6000, 15, 10, -1),
        (7000, 16, 10, -1),
        (8000, 17, 10, 1),
        (9000, 18, 10, 1),
    ], dtype=dtype)
    return events


if __name__ == "__main__":
    # 模拟事件数据
    events = generate_sample_events()

    # 图像大小
    H, W = 32, 32

    # 构建时间表面
    T_pos, T_neg = build_time_surfaces(events, H, W)

    # 计算梯度
    grad_pos_x, grad_pos_y = compute_gradients(T_pos)
    grad_neg_x, grad_neg_y = compute_gradients(T_neg)

    # 可视化结果
    show_time_surfaces(T_pos, T_neg, grad_pos_x, grad_pos_y, grad_neg_x, grad_neg_y)
