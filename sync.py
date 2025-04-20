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

def setup_camera(serial, cam_mode):
    # 打印相机信息
    print(f"\n🚀 Starting camera {serial} in mode: {cam_mode.upper()}")

    try:
        # 连接设备
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
    # 设置为 master / slave 模式
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
    # 创建显示窗口，显示事件图像
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()
    title = f"Metavision - {cam_mode.upper()} ({serial})"
    # 从设备读取事件流。
    with MTWindow(title=title, width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        # 设置键盘回调函数
        def keyboard_cb(key, scancode, action, mods):
            if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)
        # 创建事件帧生成器
        frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=300, palette=ColorPalette.CoolWarm)
        # 设置帧输出回调函数
        def on_frame_cb(ts, frame):
            window.show_async(frame)

        frame_gen.set_output_callback(on_frame_cb)

        # 事件处理主循环
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            # 把当前时间段的事件丢进 frame_gen 聚合成帧。
            frame_gen.process_events(evs)
            if window.should_close():
                break

def main():
    threads = []
    # lc = threading.Thread(target=setup_camera, args=("00051195","slave"))
    # lc.start()
    # threads.append(lc)
    # time.sleep(1)  # 小延迟以防初始化冲突
    # rc = threading.Thread(target=setup_camera, args=("00051197","master"))
    # rc.start()
    # threads.append(rc)
    #
    # for lc in threads:
    #     lc.join()
    #     rc.join()


    for config in CAMERA_CONFIGS:
        t = threading.Thread(target=setup_camera, args=(config["serial"], config["mode"]))
        t.start()
        threads.append(t)
        time.sleep(1)  # 小延迟以防初始化冲突

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
