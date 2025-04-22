import sys
import time
import threading
import torch
import numpy as np
import cv2

sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

from models.model import HeatmapUNet
from models.config import IMG_SIZE

CAMERA_CONFIGS = [
    {"serial": "00051195", "mode": "slave"},
    {"serial": "00051197", "mode": "master"},
]

def extract_peak_coords(heatmap_tensor):
    heatmap_np = heatmap_tensor.squeeze(0).detach().cpu().numpy()
    coords = []
    for hm in heatmap_np:
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords.append((x, y))
    return coords

def setup_camera(serial, cam_mode):
    print(f"\n🚀 Starting camera {serial} in mode: {cam_mode.upper()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        hw = initiate_device(path=serial)
        print(f"✅ Connected to device {serial}")
    except Exception as e:
        print(f"❌ Could not initiate device {serial}: {e}")
        return

    sync_iface = hw.get_i_camera_synchronization()
    if not sync_iface:
        print("❌ Device does not support synchronization interface.")
        return
    try:
        if cam_mode == "master":
            sync_iface.set_mode_master()
            print("✅ Set to MASTER mode.")
        else:
            sync_iface.set_mode_slave()
            print("✅ Set to SLAVE mode.")
    except Exception as e:
        print(f"❌ Failed to set {cam_mode} mode: {e}")
        return

    # 加载模型
    model = HeatmapUNet(num_keypoints=8).to(device)
    model.load_state_dict(torch.load("/home/wangzhe/ICRA2025/MY/models/checkpoints/best_model.pt", map_location=device))
    model.eval()

    iterator = EventsIterator.from_device(device=hw)
    height, width = iterator.get_size()

    frame_gen = PeriodicFrameGenerationAlgorithm(width, height, fps=100, palette=ColorPalette.CoolWarm)

    title = f"Metavision - {cam_mode.upper()} ({serial})"
    with MTWindow(title=title, width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key in [UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q]:
                window.set_close_flag()
        window.set_keyboard_callback(keyboard_cb)

        def on_frame_cb(ts, frame):
            frame_disp = frame.copy()
            resized = cv2.resize(frame_disp, (IMG_SIZE, IMG_SIZE))
            img_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                pred_heatmap = model(img_tensor)[0].cpu()

            coords = extract_peak_coords(pred_heatmap.unsqueeze(0))
            for i, (x, y) in enumerate(coords):
                x_disp = int(x * frame.shape[1] / IMG_SIZE)
                y_disp = int(y * frame.shape[0] / IMG_SIZE)
                cv2.circle(frame_disp, (x_disp, y_disp), 4, (0, 255, 0), -1)
                cv2.putText(frame_disp, str(i), (x_disp+2, y_disp-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            window.show_async(frame_disp)

        frame_gen.set_output_callback(on_frame_cb)

        for evs in iterator:
            EventLoop.poll_and_dispatch()
            frame_gen.process_events(evs)
            if window.should_close():
                break

def main():
    threads = []
    for config in CAMERA_CONFIGS:
        t = threading.Thread(target=setup_camera, args=(config["serial"], config["mode"]))
        t.start()
        threads.append(t)
        time.sleep(1)  # 避免初始化冲突

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
