from brain.sensors.lidar import LidarReader
from brain.models.snn_model import SNNModel

# Load mock LIDAR
lidar = LidarReader(use_mock=True)
data = lidar.read()
print("Sensor Data:", data)

# Run through model
model = SNNModel(neurons=128)
output = model.forward(data)
print("Model Output:", output)
