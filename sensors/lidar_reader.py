import numpy as np

def read_lidar():
    # Replace with actual LiDAR library read
    return np.random.rand(360).tolist()  # 360 degrees LiDAR

def read_distance():
    return np.random.rand(1)[0]  # distance sensor mock

def read_imu():
    return np.random.rand(1)[0]  # IMU angle mock

def get_sensor_vector():
    lidar = read_lidar()
    distance = read_distance()
    return np.array(lidar + [distance], dtype=np.float32)
