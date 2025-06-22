import numpy as np

def build_time_surfaces(events, height, width):
    """
    构建正负极性时间表面（Time Surface）。

    参数:
        events (np.ndarray): 事件数据，结构字段应包含 ('t', 'x', 'y', 'p')
        height (int): 图像高度
        width (int): 图像宽度

    返回:
        T_pos (np.ndarray): 正极性时间表面 (H×W)
        T_neg (np.ndarray): 负极性时间表面 (H×W)
    """
    T_pos = np.zeros((height, width), dtype=np.int64)
    T_neg = np.zeros((height, width), dtype=np.int64)

    for ev in events:
        t, x, y, p = ev['t'], ev['x'], ev['y'], ev['p']
        if x >= width or y >= height:
            continue  # 防止索引越界
        if p > 0:
            T_pos[y, x] = t
        else:
            T_neg[y, x] = t

    return T_pos, T_neg
