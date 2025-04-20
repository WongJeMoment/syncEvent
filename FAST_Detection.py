#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# Metavision 双机实时渲染（含结构角点检测）
# - 缩放加速、边缘降噪、Canny 自适应
# - 新增：轮廓拟合角点检测（用于红圈角点提取）
# ============================================================

import sys
import time
import threading
import numpy as np
import cv2

sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

# ----------- 相机配置 -----------
CAMERA_CONFIGS = [
    {"serial": "00051195", "mode": "slave"},   # 先开
    {"serial": "00051197", "mode": "master"},  # 后开
]

# ----------- 超参数 ------------
SCALE         = 0.5
GAUSS_KERNEL  = (5, 5)
SOBEL_THRESH  = 40
OPEN_KERNEL   = np.ones((3, 3), np.uint8)
CLOSE_KERNEL  = np.ones((3, 3), np.uint8)

# ============================================================

def auto_canny(img, sigma=0.33):
    """根据中位数自动设 Canny 阈值"""
    med = np.median(img)
    low = int(max(0, (1.0 - sigma) * med))
    high = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(img, low, high)

def detect_structure_corners(edge_img):
    """从边缘图中提取规则结构角点"""
    corners = []
    contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if 4 <= len(approx) <= 10:
            for point in approx:
                corners.append(tuple(point[0]))
    return corners

# ============================================================

def setup_camera(serial: str, cam_mode: str):
    print(f"\n🚀 Starting camera {serial} in mode: {cam_mode.upper()}")

    try:
        device = initiate_device(path=serial)
    except Exception as e:
        print(f"❌ Could not initiate device {serial}: {e}")
        return

    try:
        print(f"✅ Connected to device with serial: {device.get_serial()}")
    except Exception:
        print("⚠️  Warning: Unable to get serial number.")

    sync_iface = device.get_i_camera_synchronization()
    if not sync_iface:
        print("❌ Device does not support synchronization interface."); return
    try:
        (sync_iface.set_mode_master() if cam_mode == "master"
         else sync_iface.set_mode_slave())
        print(f"✅ Set to {cam_mode.upper()} mode.")
    except Exception as e:
        print(f"❌ Failed to set {cam_mode} mode: {e}"); return

    mv_it = EventsIterator.from_device(device=device)
    h, w = mv_it.get_size()
    win_title = f"Metavision - {cam_mode.upper()} ({serial})"

    with MTWindow(title=win_title, width=w, height=h,
                  mode=BaseWindow.RenderMode.BGR) as window:

        window.set_keyboard_callback(
            lambda k, s, a, m:
            window.set_close_flag() if k in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q) else None
        )

        frame_gen = PeriodicFrameGenerationAlgorithm(w, h, fps=300,
                                                     palette=ColorPalette.CoolWarm)

        def on_frame_cb(ts, frame):
            # --- 1. 灰度 & 下采样 ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=SCALE, fy=SCALE,
                               interpolation=cv2.INTER_AREA)

            # --- 2. 预平滑 ---
            blur = cv2.GaussianBlur(small, GAUSS_KERNEL, 0)

            # --- 3. 自适应 Canny（红）---
            canny = auto_canny(blur)
            canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, OPEN_KERNEL, 1)
            canny_up = cv2.resize(canny, (w, h), interpolation=cv2.INTER_NEAREST)

            # --- 4. Sobel（绿）---
            sx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
            sy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
            smag = cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                                   cv2.convertScaleAbs(sy), 0.5, 0)
            _, sb = cv2.threshold(smag, SOBEL_THRESH, 255, cv2.THRESH_BINARY)
            sb = cv2.morphologyEx(sb, cv2.MORPH_CLOSE, CLOSE_KERNEL, 1)
            sb = cv2.resize(sb, (w, h), interpolation=cv2.INTER_NEAREST)

            # --- 5. 检测结构角点 ---
            structure_corners = detect_structure_corners(canny_up)

            # --- 6. 叠加显示 ---
            vis = np.zeros_like(frame)
            vis[:, :, 2] = canny_up      # 红色通道：Canny
            vis[:, :, 1] = sb            # 绿色通道：Sobel
            blend = cv2.addWeighted(frame, 0.8, vis, 1.0, 0)

            # --- 7. 绘制角点 ---
            for (x, y) in structure_corners:
                cv2.circle(blend, (x, y), 5, (0, 255, 255), -1)  # 黄色圆点

            window.show_async(blend)

        frame_gen.set_output_callback(on_frame_cb)

        for evs in mv_it:
            EventLoop.poll_and_dispatch()
            frame_gen.process_events(evs)
            if window.should_close():
                break

# ============================================================

def main():
    ths = []
    # 启动 slave
    ths.append(threading.Thread(target=setup_camera,
                                args=(CAMERA_CONFIGS[0]["serial"],
                                      CAMERA_CONFIGS[0]["mode"])))
    ths[-1].start()
    time.sleep(1)

    # 启动 master
    ths.append(threading.Thread(target=setup_camera,
                                args=(CAMERA_CONFIGS[1]["serial"],
                                      CAMERA_CONFIGS[1]["mode"])))
    ths[-1].start()

    for t in ths:
        t.join()

# ============================================================

if __name__ == "__main__":
    main()
