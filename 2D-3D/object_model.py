import numpy as np
import json
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_keypoints_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    keypoints = np.array([[d["x"], d["y"], d["z"]] for d in data], dtype=np.float32)
    ids = [d["id"] for d in data]
    print(f"📦 Keypoints loaded from {json_path}")
    return keypoints, ids

def visualize_stl_and_keypoints(file_path, keypoints, ids, unit="mm"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    your_mesh = mesh.Mesh.from_file(file_path)
    for triangle in your_mesh.vectors:
        tri = np.vstack((triangle, triangle[0]))
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color='gray', linewidth=0.5)

    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], color='red', s=50)
    for idx, (x, y, z) in zip(ids, keypoints):
        ax.text(x, y, z, f'{idx}', color='blue')

    # ✅ 设置坐标轴标签加单位
    ax.set_xlabel(f'X ({unit})')
    ax.set_ylabel(f'Y ({unit})')
    ax.set_zlabel(f'Z ({unit})')

    # ✅ 标题中也加上单位
    ax.set_title(f"STL Model with JSON Keypoints (Unit: {unit})")
    plt.show()

if __name__ == "__main__":
    stl_path = "/home/wangzhe/ICRA2025/MY/STL/cube/cube.STL"  # <- 替换成你的 STL 文件路径
    json_path = "/home/wangzhe/ICRA2025/MY/STL/cube/cube.json"  # <- 替换成你的 JSON 路径

    keypoints, ids = load_keypoints_from_json(json_path)
    visualize_stl_and_keypoints(stl_path, keypoints, ids)
