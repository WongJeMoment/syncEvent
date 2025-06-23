import numpy as np
import cv2
import os
from tqdm import tqdm

def npy_to_color_video(npy_path, video_path, resolution=(640, 480), fps=30):
    print(f"📂 Loading events from: {npy_path}")
    events = np.load(npy_path)

    width, height = resolution
    duration = (events['t'][-1] - events['t'][0]) / 1e6  # 微秒转秒
    total_frames = int(duration * fps)
    print(f"🎥 Duration: {duration:.2f}s, Total frames: {total_frames}")

    frame_interval = (events['t'][-1] - events['t'][0]) / total_frames
    start_time = events['t'][0]

    # 彩色视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, resolution, isColor=True)

    for i in tqdm(range(total_frames), desc="Converting to color video"):
        frame_start = start_time + i * frame_interval
        frame_end = frame_start + frame_interval

        mask = (events['t'] >= frame_start) & (events['t'] < frame_end)
        evs = events[mask]

        # 创建背景为浅灰色的彩色帧
        frame = np.full((height, width, 3), 220, dtype=np.uint8)

        for e in evs:
            x, y, p = e['x'], e['y'], e['p']
            if x < width and y < height:
                if p > 0:
                    frame[y, x] = [0, 0, 255]   # 红色（正极性）
                else:
                    frame[y, x] = [255, 0, 0]   # 蓝色（负极性）

        # 在左上角添加当前帧的最小时间戳信息（单位：秒）
        if evs.size > 0:
            min_t = evs['t'].min() / 1e6  # 转为秒
            time_text = f"Time: {min_t:.6f}s"
            cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2, cv2.LINE_AA)  # 黑色文字描边
            cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 1, cv2.LINE_AA)  # 白色文字

        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Color video saved to: {video_path}")

if __name__ == "__main__":
    input_npy = "/home/wangzhe/ICRA2025/MY/EventTxtData/cropped_events.npy"
    output_video = "/home/wangzhe/ICRA2025/MY/EventTxtData/events.avi"
    resolution = (1920, 1080)  # 替换为你的相机分辨率
    fps = 300

    npy_to_color_video(input_npy, output_video, resolution, fps)
