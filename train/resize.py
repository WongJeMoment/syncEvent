import os
import cv2

# 输入原始图像文件夹
src_folder = '/home/wangzhe/ICRA2025/MY/train/frame'
# 输出调整尺寸后的图像文件夹
dst_folder = '/home/wangzhe/ICRA2025/MY/train/frame_resized'
os.makedirs(dst_folder, exist_ok=True)

# 设置目标尺寸（必须和热图脚本中一致）
TARGET_WIDTH = 256
TARGET_HEIGHT = 256

for fname in os.listdir(src_folder):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    src_path = os.path.join(src_folder, fname)
    dst_path = os.path.join(dst_folder, fname)

    img = cv2.imread(src_path)
    if img is None:
        print(f"❌ Failed to read image: {src_path}")
        continue

    resized_img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    cv2.imwrite(dst_path, resized_img)
    print(f"✅ Saved resized image: {dst_path}")
