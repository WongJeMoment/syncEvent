import sys
import time
import threading
import os
import cv2

sys.path.append("/usr/lib/python3/dist-packages/")
# save video
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

# 配置两个设备的信息
CAMERA_CONFIGS = [
    {"serial": "00051195", "mode": "slave"},
    {"serial": "00051197", "mode": "master"},
]

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

    video_writer = None
    video_path = f"{cam_mode}_{serial}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    with MTWindow(title=title, width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)

        frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=300, palette=ColorPalette.CoolWarm)

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
            frame_gen.process_events(evs)
            if window.should_close():
                break

    if video_writer:
        video_writer.release()
        print(f"🎞️ Video saved: {video_path}")

def main():
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
