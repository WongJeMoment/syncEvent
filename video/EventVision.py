import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_event_paper_style(npy_path, sample_ratio=0.001):  # 稀疏化
    events = np.load(npy_path, allow_pickle=True)
    N = len(events)
    indices = np.random.choice(N, size=int(N * sample_ratio), replace=False)
    sampled = events[indices]

    x, y, t, p = sampled['x'], sampled['y'], sampled['t'], sampled['p']
    t = (t - t.min()) / 1e6

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[p > 0], t[p > 0], y[p > 0], c='red', s=3, alpha=0.6)
    ax.scatter(x[p <= 0], t[p <= 0], y[p <= 0], c='blue', s=3, alpha=0.6)

    # 标签样式
    label_font = {'fontsize': 14, 'labelpad': 12, 'weight': 'bold'}
    tick_font = {'labelsize': 12}

    ax.set_xlabel("X", **label_font)
    ax.set_ylabel("Time", **label_font)   # 现在 T 在 Y 轴
    ax.set_zlabel("Y", **label_font)      # Y 在 Z 轴
    ax.tick_params(**tick_font)

    # 视角
    ax.view_init(elev=30, azim=-60)

    # 网格
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 背景与线条
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_color((0.5, 0.5, 0.5, 0.3))

    plt.tight_layout()
    plt.show()

# 示例调用
plot_event_paper_style("/home/wangzhe/ICRA2025/MY/DatasetRotation/Part3/120/slave_00051195_events.npy")
