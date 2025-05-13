import cv2

def get_image_size_info(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 无法读取图像")
        return
    h, w, c = img.shape
    print(f"图像路径: {image_path}")
    print(f"宽度 (width):  {w}")
    print(f"高度 (height): {h}")
    print(f"通道数 (channels): {c}")
    print(f"尺寸 (H x W x C): {img.shape}")

# 示例使用
get_image_size_info("/home/wangzhe/ICRA2025/MY/data/part2_train_frame/1.jpg")
