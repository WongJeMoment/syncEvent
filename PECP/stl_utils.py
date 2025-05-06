from stl import mesh
import numpy as np

def load_stl_contour_points(stl_path, sample_num=1000):
    """
    载入 STL 文件，并采样 M 个轮廓点
    """
    model = mesh.Mesh.from_file(stl_path)
    all_points = model.vectors.reshape(-1, 3)
    indices = np.random.choice(len(all_points), sample_num, replace=False)
    return all_points[indices]
