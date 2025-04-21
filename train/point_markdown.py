import cv2
import json
import os

img_folder = '/home/wangzhe/ICRA2025/MY/train/frame'  # 图像文件夹
json_output_dir = '/home/wangzhe/ICRA2025/MY/train/json'  # 每张图一个JSON文件
os.makedirs(json_output_dir, exist_ok=True)

# 获取所有图像
img_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    filename_wo_ext = os.path.splitext(img_file)[0]
    output_json_path = os.path.join(json_output_dir, f'{filename_wo_ext}.json')

    keypoints = []
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图像：{img_path}")
        continue

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            keypoints.append([x, y])
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow("Image", img)
            print(f"📌 Point {len(keypoints)}: ({x}, {y})")

    print(f"\n🖼️ 当前图像: {img_file}，点击选点，按任意键保存并继续")
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 写入对应的JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(keypoints, f, indent=4)
    print(f"✅ {len(keypoints)} 个点保存至 {output_json_path}")
