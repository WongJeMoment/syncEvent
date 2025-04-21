import cv2
import os

# === 参数设置 ===
video_path = '/home/wangzhe/ICRA2025/MY/video/slave_00051195.avi'        # 替换为你的avi视频路径
output_dir = '/home/wangzhe/ICRA2025/MY/video/frame2'         # 输出帧的保存文件夹
os.makedirs(output_dir, exist_ok=True)

# === 打开视频文件 ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 无法打开视频文件！")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ 视频帧提取完成！")
        break

    # 保存每一帧为jpg图像
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
    cv2.imwrite(frame_filename, frame)
    print(f'🖼️ 保存帧：{frame_filename}')
    frame_count += 1

cap.release()
