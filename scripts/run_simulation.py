from brain.sensors.lidar import LidarReader

lidar = LidarReader(use_mock=True)
data = lidar.read()
print("Sensor Data:", data)
