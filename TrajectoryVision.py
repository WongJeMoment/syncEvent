import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
df = pd.read_csv("/home/wangzhe/ICRA2025/MY/pose_output.csv")

# 提取轨迹点
x, y, z = df['Tx'], df['Ty'], df['Tz']

# 三维轨迹可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label="Trajectory", linewidth=2)
ax.set_xlabel("X (camera)")
ax.set_ylabel("Y (camera)")
ax.set_zlabel("Z (camera)")
ax.set_title("Object Trajectory (Camera Frame)")
plt.legend()
plt.tight_layout()
plt.show()
