import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_event_3d(npy_path, sample_ratio=0.01):
    print(f"📂 Loading events from: {npy_path}")
    events = np.load(npy_path)

    # 随机采样部分数据以减少渲染压力
    N = len(events)
    indices = np.random.choice(N, size=int(N * sample_ratio), replace=False)
    sampled = events[indices]

    # 获取坐标与极性
    x, y, t, p = sampled['x'], sampled['y'], sampled['t'], sampled['p']
    t = (t - t.min()) / 1e6  # 微秒转秒，归一化时间

    # 绘图
    fig = plt.figure(figsize=(16, 9))  # 图像 16:9
    ax = fig.add_subplot(111, projection='3d')

    # 设置 3D 坐标轴的视觉比例为 16:9:9（X:Y:Z）
    ax.set_box_aspect([16, 9, 9])

    # 区分正负极性
    pos_mask = p > 0
    neg_mask = ~pos_mask

    # 绘制散点图（X轴: 时间, Y轴: y, Z轴: x）
    ax.scatter(t[pos_mask], y[pos_mask], x[pos_mask], c='red', s=0.1, alpha=0.6)
    ax.scatter(t[neg_mask], y[neg_mask], x[neg_mask], c='blue', s=0.1, alpha=0.6)

    # 标签与视角
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.view_init(elev=20, azim=-60)  # 可调视觉角度

    plt.tight_layout()
    plt.show()

# === 使用示例 ===
if __name__ == "__main__":
    npy_file_path = "output/CubeVideo/Rotation40rpm/master_00051197_events.npy"
    plot_event_3d(npy_file_path)
