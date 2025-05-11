import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# ✅ 设置中文字体支持，避免 matplotlib 中文乱码警告
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def read_3d_model(file_path):
    mesh = trimesh.load(file_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    return mesh

def visualize_model(mesh, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    faces = mesh.faces

    # 绘制模型表面
    mesh_faces = [vertices[face] for face in faces]
    mesh_collection = Poly3DCollection(mesh_faces, alpha=0.6)
    mesh_collection.set_facecolor([0.6, 0.8, 1.0])  # 设置面颜色（淡蓝）

    ax.add_collection3d(mesh_collection)

    # 设置坐标轴范围
    xyz_min = vertices.min(axis=0)
    xyz_max = vertices.max(axis=0)
    margin = 0.1 * (xyz_max - xyz_min).max()
    ax.set_xlim([xyz_min[0] - margin, xyz_max[0] + margin])
    ax.set_ylim([xyz_min[1] - margin, xyz_max[1] + margin])
    ax.set_zlim([xyz_min[2] - margin, xyz_max[2] + margin])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("STL模型渲染")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ 图像已保存: {save_path}")
    plt.show()

def main():
    stl_path = "/home/wangzhe/ICRA2025/MY/STL/PART2/Part2.STL"  # ✅ STL文件路径
    save_img_path = "/home/wangzhe/ICRA2025/MY/rendered_stl.png"  # ✅ 保存渲染图像

    if not os.path.exists(stl_path):
        print(f"❌ 文件不存在: {stl_path}")
        return

    mesh = read_3d_model(stl_path)
    visualize_model(mesh, save_path=save_img_path)

if __name__ == "__main__":
    main()
