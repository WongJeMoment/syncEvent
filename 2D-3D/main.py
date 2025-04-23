# keypoint_generator/main.py

from hash_table import build_hash_table
from frequency_counter import count_semantic_point_frequency
from keypoint_selector import select_top_k_keypoints

if __name__ == "__main__":
    # 示例数据（你可以替换为从数据集中读取）
    uv_list = [(10, 20), (30, 40), (50, 60), (70, 80)]
    xyz_list = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]

    image_annotations = [
        [(10, 20), (30, 40)],
        [(30, 40), (50, 60)],
        [(10, 20), (70, 80)],
        [(30, 40)],
    ]

    # 运行流程
    hash_table = build_hash_table(uv_list, xyz_list)
    freq_counter = count_semantic_point_frequency(image_annotations)
    top_keypoints_3d = select_top_k_keypoints(freq_counter, hash_table, k=3)

    print("🔑 Final selected 3D keypoints:")
    for pt in top_keypoints_3d:
        print(pt)
