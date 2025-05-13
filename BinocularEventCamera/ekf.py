# core/ekf.py
import numpy as np

class EKFPoint:
    def __init__(self, init_xyz, dt=0.01):
        self.x = np.hstack([init_xyz, [0, 0, 0]])  # 状态向量 [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1.0                  # 协方差矩阵
        self.Q = np.eye(6) * 1e-4                 # 过程噪声
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)                # 观测矩阵
        self.dt = dt
        self._build_F(dt)

    def _build_F(self, dt):
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

    def predict(self, dt=None):
        if dt is not None:
            self.dt = dt
            self._build_F(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R=None):
        if R is None:
            R = np.eye(3) * 0.01
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P