import numpy as np
import cv2

class PointKalman:
    def __init__(self, init_pt, dt=1.0):
        self.dt = dt
        self.kf = cv2.KalmanFilter(6, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.zeros((2, 6), np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1

        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        self.kf.statePost[0, 0] = init_pt[0]
        self.kf.statePost[1, 0] = init_pt[1]

        self.base_Q = np.diag([1e-2, 1e-2, 1e-1, 1e-1, 1.0, 1.0]).astype(np.float32)
        self.kf.processNoiseCov = self.base_Q.copy()
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def predict(self):
        return self.kf.predict()[:2].flatten()

    def correct(self, pt):
        measurement = np.array([[pt[0]], [pt[1]]], dtype=np.float32)
        self.kf.correct(measurement)

    def set_process_noise_scale(self, scale):
        scale = np.clip(scale, 0.5, 10.0)
        self.kf.processNoiseCov = self.base_Q * scale

    def set_measurement_noise_scale(self, scale):
        """动态调整观测噪声协方差 R（越小表示越信任观测）"""
        self.kf.measurementNoiseCov = self.base_R * scale
