# keypoint_generator/frequency_counter.py
from collections import Counter

def count_semantic_point_frequency(image_annotations):
    """
    输入：每张图片的标注2D点集合列表
    输出：统计每个点的出现频率
    """
    freq_counter = Counter()
    for annotated_uvs in image_annotations:
        for uv in annotated_uvs:
            freq_counter[tuple(uv)] += 1
    return freq_counter
