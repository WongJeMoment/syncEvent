import numpy as np
import cv2
import scipy.ndimage
def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img

def pad_to_multiple(img, divisor=32):
    h, w = img.shape[:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return img_padded, (h, w)

def extract_peak_coords(heatmap_tensor, threshold=0.15, nms_radius=5, merge_distance=15, top_k=14, orig_size=None):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    raw_coords = []
    for hm in heatmap_np:
        neighborhood = (nms_radius * 2) + 1
        local_max = (hm == scipy.ndimage.maximum_filter(hm, size=neighborhood))
        mask = (hm > threshold) & local_max
        ys, xs = np.where(mask)
        keypoints = [(x, y, hm[y, x]) for x, y in zip(xs, ys)]
        raw_coords.extend(keypoints)
    raw_coords.sort(key=lambda k: k[2], reverse=True)
    final_coords = []
    visited = np.zeros(len(raw_coords), dtype=bool)
    for i, (x_i, y_i, score_i) in enumerate(raw_coords):
        if visited[i]:
            continue
        final_coords.append((x_i, y_i, score_i))
        for j, (x_j, y_j, _) in enumerate(raw_coords):
            if not visited[j] and np.hypot(x_i - x_j, y_i - y_j) < merge_distance:
                visited[j] = True
    if len(final_coords) > top_k:
        final_coords = final_coords[:top_k]
    coords = [(x, y) for (x, y, _) in final_coords]
    if orig_size is not None:
        h_hm, w_hm = heatmap_np.shape[1:]
        scale_x = orig_size[1] / w_hm
        scale_y = orig_size[0] / h_hm
        coords = [(int(x * scale_x), int(y * scale_y)) for (x, y) in coords]
    return coords


def draw_keypoints_only(img, keypoints, radius=5, color=(0, 255, 255), draw_index=True, font_scale=0.5):
    """
    在图像上绘制关键点和编号，编号使用黑底白字，适合浅背景。

    参数:
        img: 输入图像
        keypoints: Nx2 数组，关键点坐标
        radius: 圆点半径
        color: 圆点颜色 (BGR)
        draw_index: 是否绘制编号
    """
    for idx, pt in enumerate(keypoints):
        x, y = pt
        if np.isnan(x) or np.isnan(y):
            continue

        # 画关键点圆圈
        cv2.circle(img, (int(x), int(y)), radius, color, -1)

        if draw_index:
            text = str(idx)
            org = (int(x) + 5, int(y) - 5)

            # 黑色描边（底层）
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            # 白色文本（上层）
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img

