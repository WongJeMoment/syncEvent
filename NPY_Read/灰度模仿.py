import numpy as np
import cv2
import os


def event_frame_browser(npy_path, resolution=(640, 480), frame_time_us=1000):
    print(f"📂 Loading events from: {npy_path}")
    events = np.load(npy_path)
    width, height = resolution

    duration_us = events['t'][-1] - events['t'][0]
    total_frames = int(duration_us // frame_time_us)
    print(f"🕒 Duration: {duration_us / 1e6:.2f}s, Total frames: {total_frames}")

    start_time = events['t'][0]
    frame_idx = 0

    while True:
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= total_frames:
            frame_idx = total_frames - 1

        frame_start = start_time + frame_idx * frame_time_us
        frame_end = frame_start + frame_time_us
        mask = (events['t'] >= frame_start) & (events['t'] < frame_end)
        evs = events[mask]

        # 灰度累加图（范围0-255）
        acc = np.full((height, width), 127, dtype=np.int16)  # 初始化为中灰色（127）
        for e in evs:
            x, y, p = int(e['x']), int(e['y']), int(e['p'])
            if 0 <= x < width and 0 <= y < height:
                acc[y, x] += 20 if p > 0 else -20

        acc = np.clip(acc, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(acc, cv2.COLOR_GRAY2BGR)

        time_str = f"Frame {frame_idx + 1}/{total_frames} | Time: {(frame_start - start_time) / 1e6:.3f}s"
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Event-like Grayscale Viewer", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('e'):
            frame_idx += 1
        elif key == ord('q'):
            frame_idx -= 1
        elif key in [27, ord('x')]:  # ESC or x
            print("👋 退出查看")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_npy = "/home/wangzhe/ICRA2025/MY/DatasetRotation/Cube/120/master_00051197_events.npy"
    resolution = (1280, 720)
    event_frame_browser(input_npy, resolution, frame_time_us=1000)  # 每帧1ms 模拟灰度
