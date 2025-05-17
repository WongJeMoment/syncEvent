import json
import numpy as np
def load_model_points_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    ids = [d["id"] for d in data]
    points = np.array([[d["x"], d["y"], d["z"]] for d in data], dtype=np.float32)
    return points, ids
