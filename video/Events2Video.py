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

        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Color video saved to: {video_path}")

if __name__ == "__main__":
    input_npy = "/home/wangzhe/ICRA2025/MY/DatasetRotation/Cube/120/master_00051197_events.npy"
    output_video = "output/master_00051197_color.avi"
    resolution = (1280, 720)  # 替换为你的相机分辨率
    fps = 300

    npy_to_color_video(input_npy, output_video, resolution, fps)
