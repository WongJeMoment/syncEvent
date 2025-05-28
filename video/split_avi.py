import cv2
import os

def video_to_frames(video_path, output_folder):
    # 检查输出文件夹，没有就给它造一个
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 创建输出文件夹: {output_folder}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ 无法打开视频文件: {video_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🚀 视频读取完毕！")
            break

        # 保存帧，命名为 frame_000001.jpg 这种
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)

        if frame_count % 50 == 0:
            print(f"✨ 已保存 {frame_count} 帧...")

        frame_count += 1

    cap.release()
    print(f"✅ 全部完成！总共保存了 {frame_count} 帧。")

if __name__ == "__main__":
    video_path = "/home/wangzhe/ICRA2025/MY/DataMoment/Cube/master_00051197.avi"   # <-- 换成你的avi视频路径
    output_folder = "/home/wangzhe/ICRA2025/MY/DataMoment/CubeImg/master" # <-- 换成你想要保存帧的文件夹
    video_to_frames(video_path, output_folder)
