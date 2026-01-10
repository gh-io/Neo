import torch
import torch.nn as nn
import serial

class NeurobotANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(361, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

ser = serial.Serial('/dev/ttyUSB0', 115200)
model = NeurobotANN()

actions = ["FORWARD", "LEFT", "RIGHT", "STOP"]

while True:
    state = torch.rand((1, 361))
    action = actions[torch.argmax(model(state))]
    ser.write((action + "\n").encode())
  
