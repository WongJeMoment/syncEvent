import numpy as np

def load_events(npy_path):
    """
    从 .npy 文件中加载事件数据。

    参数:
        npy_path (str): 事件数据的文件路径，文件应为结构化数组，字段包括 t, x, y, p。

    返回:
        np.ndarray: 包含结构字段 ('t', 'x', 'y', 'p') 的事件数组
    """
    try:
        events = np.load(npy_path)
        assert {'t', 'x', 'y', 'p'}.issubset(events.dtype.names), "字段应包含 't', 'x', 'y', 'p'"
        return events
    except Exception as e:
        print(f"[ERROR] 事件加载失败: {e}")
        return None
