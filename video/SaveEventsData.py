import sys
import time
import threading
import os
import cv2
import numpy as np

sys.path.append("/usr/lib/python3/dist-packages/")

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

# 配置两个设备的信息
CAMERA_CONFIGS = [
    {"serial": "00051195", "mode": "slave"},
    {"serial": "00051197", "mode": "master"},
]

# 每次最大处理的事件数（但不丢弃事件，只是分批处理）
MAX_EVENTS = 100000

def setup_camera(serial, cam_mode):
    print(f"\n🚀 Starting camera {serial} in mode: {cam_mode.upper()}")

    try:
        device = initiate_device(path=serial)
    except Exception as e:
        print(f"❌ Could not initiate device {serial}: {e}")
        return

    try:
        print(f"✅ Connected to device with serial: {device.get_serial()}")
    except:
        print("⚠️  Warning: Unable to get serial number.")

    sync_iface = device.get_i_camera_synchronization()
    if not sync_iface:
        print("❌ Device does not support synchronization interface.")
        return

    try:
        if cam_mode == "master":
            sync_iface.set_mode_master()
            print("✅ Set to MASTER mode.")
            print("ℹ️  Reminder: Start slave first, then start this master.")
        else:
            sync_iface.set_mode_slave()
            print("✅ Set to SLAVE mode.")
            print("⏳ Waiting for master sync signal to start...")
    except Exception as e:
        print(f"❌ Failed to set {cam_mode} mode: {e}")
        return

    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()
    title = f"Metavision - {cam_mode.upper()} ({serial})"

    # ✅ 创建输出文件夹（如不存在）
    os.makedirs("output", exist_ok=True)

    video_writer = None
    video_path = os.path.join("output", f"{cam_mode}_{serial}.avi")
    npy_path = os.path.join("output", f"{cam_mode}_{serial}_events.npy")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # ✅ 准备保存事件数据
    all_x, all_y, all_t, all_p = [], [], [], []

    with MTWindow(title=title, width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)

        frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=30, palette=ColorPalette.CoolWarm)

        def on_frame_cb(ts, frame):
            nonlocal video_writer
            window.show_async(frame)
            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            video_writer.write(frame)

        frame_gen.set_output_callback(on_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()

            # ✅ 分批处理事件，同时保留全部事件
            for i in range(0, len(evs), MAX_EVENTS):
                batch = evs[i:i+MAX_EVENTS]
                frame_gen.process_events(batch)

                # ✅ 收集事件字段
                all_x.append(batch['x'])
                all_y.append(batch['y'])
                all_t.append(batch['t'])
                all_p.append(batch['p'].astype(np.int8))  # polarity 转为 int8

            if window.should_close():
                break

    if video_writer:
        video_writer.release()
        print(f"🎞️ Video saved: {video_path}")

    # ✅ 保存为 .npy
    if all_x:
        try:
            x = np.concatenate(all_x)
            y = np.concatenate(all_y)
            t = np.concatenate(all_t)
            p = np.concatenate(all_p)

            structured_events = np.zeros(len(x), dtype=[('x', 'u2'), ('y', 'u2'), ('t', 'u8'), ('p', 'i1')])
            structured_events['x'] = x
            structured_events['y'] = y
            structured_events['t'] = t
            structured_events['p'] = p

            np.save(npy_path, structured_events)
            print(f"💾 Events saved as .npy: {npy_path}")
        except Exception as e:
            print(f"❌ Failed to save events for {serial}: {e}")

def main():
    threads = []

    for config in CAMERA_CONFIGS:
        t = threading.Thread(target=setup_camera, args=(config["serial"], config["mode"]))
        t.start()
        threads.append(t)
        time.sleep(1)  # 小延迟保证先启动 slave

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
