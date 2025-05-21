import sys
import time
import threading
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

# 每批最多处理的事件数
MAX_EVENTS = 300000

def setup_camera(serial, cam_mode):
    print(f"\n🚀 Starting camera {serial} in mode: {cam_mode.upper()}")

    try:
        device = initiate_device(path=serial)
    except Exception as e:
        return

    sync_iface = device.get_i_camera_synchronization()
    if not sync_iface:
        return

    try:
        if cam_mode == "master":
            sync_iface.set_mode_master()
            print(f"[{serial}] ✅ Set to MASTER mode.")
            print(f"[{serial}] ℹ️  Reminder: Start slave first, then start this master.")
        else:
            sync_iface.set_mode_slave()
            print(f"[{serial}] ✅ Set to SLAVE mode.")
            print(f"[{serial}] ⏳ Waiting for master sync signal to start...")
    except Exception as e:
        return

    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()
    title = f"Metavision - {cam_mode.upper()} ({serial})"

    with MTWindow(title=title, width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=300, palette=ColorPalette.CoolWarm)

        def on_frame_cb(ts, frame):
            window.show_async(frame)

        frame_gen.set_output_callback(on_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()

            # 限制事件数量
            if len(evs) > MAX_EVENTS:
                evs = evs[:MAX_EVENTS]

            frame_gen.process_events(evs)

            if window.should_close():
                break

def main():
    threads = []

    for config in CAMERA_CONFIGS:
        t = threading.Thread(target=setup_camera, args=(config["serial"], config["mode"]))
        t.start()
        threads.append(t)
        time.sleep(1)  # 小延迟避免同步冲突

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
