import trimesh
import pyrender
import matplotlib.pyplot as plt
import numpy as np
# 加载 OBJ 或 STL
mesh = trimesh.load('/home/wangzhe/ICRA2025/MY/STL/cube/cube1.obj')  # 支持 .obj/.stl

# 设置摄像机
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
scene.add(camera, pose=np.eye(4))  # 可自定义视角姿态

# 渲染图像
r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)

plt.imshow(color)
plt.title("Rendered Projection")
plt.axis('off')
plt.show()
