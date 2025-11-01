class StrategicANN(nn.Module):
    def __init__(self):
        super(StrategicANN, self).__init__()
        # Vision CNN
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 64)
        
        # LiDAR + Distance + IMU sensors
        self.sensor_fc1 = nn.Linear(361, 64)  # 360 LiDAR + 1 distance
        self.sensor_fc2 = nn.Linear(64, 32)
        
        # Combined decision layer
        self.fc1 = nn.Linear(64+32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)  # Actions: FORWARD, LEFT, RIGHT, BACKWARD

    def forward(self, image, sensors):
        vision_feat = self.cnn(image)
        x = self.sensor_fc1(sensors)
        x = self.relu(x)
        sensor_feat = self.sensor_fc2(x)
        sensor_feat = self.relu(sensor_feat)
        
        combined = torch.cat((vision_feat, sensor_feat), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        output = self.fc2(x)
        return output
