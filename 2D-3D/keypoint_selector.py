# keypoint_generator/keypoint_selector.py

def select_top_k_keypoints(freq_counter, hash_table, k=8):
    """
    根据频率 + 哈希表，选出Top-K 3D关键点
    """
    top_k_uvs = [uv for uv, _ in freq_counter.most_common(k)]
    keypoints_3d = [hash_table[uv] for uv in top_k_uvs if uv in hash_table]
    return keypoints_3d
