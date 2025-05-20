import numpy as np
import cv2
import os
from tqdm import tqdm  # ✅ 添加进度条支持

def npy_to_video(npy_path, video_path, resolution=(640, 480), fps=30):
    print(f"📂 Loading events from: {npy_path}")
    events = np.load(npy_path)

    width, height = resolution
    duration = (events['t'][-1] - events['t'][0]) / 1e6  # 微秒 -> 秒
    total_frames = int(duration * fps)
    print(f"🎥 Duration: {duration:.2f}s, Total frames: {total_frames}")

    frame_interval = (events['t'][-1] - events['t'][0]) / total_frames
    start_time = events['t'][0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, resolution, isColor=False)

    for i in tqdm(range(total_frames), desc="Converting to video"):
        frame_start = start_time + i * frame_interval
        frame_end = frame_start + frame_interval

        mask = (events['t'] >= frame_start) & (events['t'] < frame_end)
        evs = events[mask]

        frame = np.zeros((height, width), dtype=np.uint8)
        for e in evs:
            x, y, p = e['x'], e['y'], e['p']
            if x < width and y < height:
                frame[y, x] = 255 if p > 0 else 127

        video_writer.write(frame)

    video_writer.release()
    print(f"✅ Video saved to: {video_path}")

if __name__ == "__main__":
    input_npy = "output/master_00051197_events.npy"
    output_video = "output/master_00051197_from_npy.avi"
    resolution = (640, 480)  # 替换为你的真实分辨率
    fps = 30

    npy_to_video(input_npy, output_video, resolution, fps)
