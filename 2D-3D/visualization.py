import cv2
import numpy as np

def draw_cube_with_keypoints(img, imgpts, keypoints):
    """
    绘制立方体框架和关键点。

    参数:
        img: 原始图像
        imgpts: 投影后的8个立方体点 (shape: [8, 2])
        keypoints: 检测到的2D关键点 (shape: [8, 2])
    返回:
        绘制后的图像
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # 画底面四边形
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)

    # 画四根立柱
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)

    # 画顶面四边形
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)

    # 画关键点
    for idx, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)  # 画黄点
        cv2.putText(img, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 255), 2)  # 写编号

    return img
