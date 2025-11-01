import numpy as np

def read_lidar():
    return np.random.rand(360).tolist()

def read_distance():
    return np.random.rand(1)[0]

def read_imu():
    return np.random.rand(1)[0]

def get_sensor_vector():
    lidar = read_lidar()
    distance = read_distance()
    return np.array(lidar + [distance], dtype=np.float32)
