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

# 相机配置
CAMERA_CONFIGS = [
    {"serial": "00051195", "mode": "slave"},
    {"serial": "00051197", "mode": "master"},
]

MAX_EVENTS = 300000
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    video_path = os.path.join(OUTPUT_DIR, f"{cam_mode}_{serial}.avi")
    npy_path = os.path.join(OUTPUT_DIR, f"{cam_mode}_{serial}_events.npy")

    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # ✅ 用字段列表代替事件对象列表
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

            if len(evs) == 0:
                continue

            if len(evs) > MAX_EVENTS:
                evs = evs[:MAX_EVENTS]
                print(f"[{serial}] ⚠️ Truncated to {MAX_EVENTS} events")

            print(f"[{serial}] Events received: {len(evs)}")

            # ✅ 收集字段到各自数组
            all_x.append(evs['x'])
            all_y.append(evs['y'])
            all_t.append(evs['t'])
            all_p.append(evs['p'].astype(np.int8))

            frame_gen.process_events(evs)

            if window.should_close():
                break

    if video_writer:
        video_writer.release()
        print(f"🎞️ Video saved: {video_path}")

    # ✅ 合并保存为 .npy
    if all_x:
        try:
            print(f"[{serial}] 🧩 Concatenating and saving event data...")
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
    else:
        print(f"[{serial}] ⚠️ No events collected — .npy not saved.")

def main():
    print(f"📁 All outputs will be saved in: {os.path.abspath(OUTPUT_DIR)}")
    threads = []

    for config in CAMERA_CONFIGS:
        t = threading.Thread(target=setup_camera, args=(config["serial"], config["mode"]))
        t.start()
        threads.append(t)
        time.sleep(1)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
