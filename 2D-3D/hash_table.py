# keypoint_generator/hash_table.py

def build_hash_table(uv_list, xyz_list):
    """
    从 (u, v) → (x, y, z) 构建哈希映射表
    """
    return {tuple(uv): tuple(xyz) for uv, xyz in zip(uv_list, xyz_list)}
