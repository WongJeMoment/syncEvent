import numpy as np

class PointKalman:
    def __init__(self, init_pt):
        self.dt = 1.0

        # 状态: [x, y, vx, vy, ax, ay]
        self.x = np.array([[init_pt[0]], [init_pt[1]], [0], [0], [0], [0]])
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * 0.2  # 提高对动态响应的敏感度
        self.R = np.eye(2) * 0.5

        dt = self.dt
        dt2 = 0.5 * dt ** 2

        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def correct(self, z):
        z = np.array([[z[0]], [z[1]]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
