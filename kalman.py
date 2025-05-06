import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_variance, measurement_variance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self):
        # 预测步骤
        self.state = np.dot(self.A, self.state)
        self.covariance = np.dot(np.dot(self.A, self.covariance), self.A.T) + self.process_variance

    def update(self, measurement):
        # 更新步骤
        innovation = measurement - np.dot(self.H, self.state)
        innovation_covariance = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.measurement_variance
        kalman_gain = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(innovation_covariance))

        self.state = self.state + np.dot(kalman_gain, innovation)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(kalman_gain, self.H)), self.covariance)

# 初始化卡尔曼滤波器
initial_state = np.array([0, 0])  # 初始状态
initial_covariance = np.eye(2)  # 初始协方差
process_variance = np.eye(2) * 0.1  # 过程方差
measurement_variance = np.eye(2) * 1  # 测量方差

kf = KalmanFilter(initial_state, initial_covariance, process_variance, measurement_variance)

# 激光雷达和相机测量的信息
radar_measurement = np.array([10, np.pi/4])  # 激光雷达测量的极坐标信息 (距离, 角度)
camera_measurement = np.array([8, 6])  # 相机测量的直角坐标信息 (x, y)

# 定义状态转移矩阵和观测矩阵
kf.A = np.eye(2)  # 状态转移矩阵
kf.H = np.array([[np.cos(radar_measurement[1]), np.sin(radar_measurement[1])], [1, 1]])  # 观测矩阵

# 执行卡尔曼滤波融合
kf.predict()
kf.update(radar_measurement)
kf.update(camera_measurement)

# 输出融合后的位置估计
print("Fused Position Estimate:", kf.state)