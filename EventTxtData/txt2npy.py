import numpy as np

def txt_to_event_npy(txt_path, npy_path):
    # 跳过前6行注释，读取后续数据
    data = np.loadtxt(txt_path, comments='#')

    # 分列读取
    t = (data[:, 0] * 1e6).astype(np.int64)  # 秒 → 微秒
    x = data[:, 1].astype(np.uint16)
    y = data[:, 2].astype(np.uint16)
    p = data[:, 3].astype(np.uint8)

    # 定义结构化 dtype
    event_dtype = np.dtype([
        ('t', np.int64),
        ('x', np.uint16),
        ('y', np.uint16),
        ('p', np.uint8)
    ])

    # 创建结构化数组
    events = np.zeros(t.shape[0], dtype=event_dtype)
    events['t'] = t
    events['x'] = x
    events['y'] = y
    events['p'] = p

    # 保存为 npy 文件
    np.save(npy_path, events)
    print(f"✅ Converted and saved to {npy_path}")


# 示例使用
txt_path = "/home/wangzhe/ICRA2025/MY/EventTxtData/events.txt"
npy_path = "/home/wangzhe/ICRA2025/MY/EventTxtData/events.npy"
txt_to_event_npy(txt_path, npy_path)
