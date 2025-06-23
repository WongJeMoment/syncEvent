import numpy as np
import os

def crop_event_npy_by_time(input_path, output_path, start_us, end_us):
    """
    从原始npy文件中裁剪出给定时间范围内的事件，并保存为新npy文件。

    Args:
        input_path (str): 原始事件npy路径
        output_path (str): 裁剪后保存路径
        start_us (int): 起始时间戳（单位：微秒）
        end_us (int): 结束时间戳（单位：微秒）
    """
    print(f"📂 Loading events from: {input_path}")
    events = np.load(input_path)
    print(f"🕒 Total event time range: [{events['t'][0]} - {events['t'][-1]}] us")

    # 时间戳范围裁剪
    mask = (events['t'] >= start_us) & (events['t'] <= end_us)
    cropped_events = events[mask]

    print(f"✂️  Cropped events: {len(cropped_events)}")
    np.save(output_path, cropped_events)
    print(f"✅ Saved cropped events to: {output_path}")

if __name__ == "__main__":
    input_npy = "/home/wangzhe/ICRA2025/MY/EventTxtData/events.npy"
    output_npy = "/home/wangzhe/ICRA2025/MY/EventTxtData/cropped_events.npy"
    start_us = 1_000_000   # 起始时间，例如 1 秒（1_000_000 微秒）
    end_us = 2_000_000     # 结束时间，例如 2 秒（2_000_000 微秒）

    crop_event_npy_by_time(input_npy, output_npy, start_us, end_us)
