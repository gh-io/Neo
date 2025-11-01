import torch
import torch.nn as nn

class ReflexSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(361, 3)  # LiDAR + distance → motor spike commands

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # 0-1 motor intensity
