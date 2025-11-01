import torch
import torch.nn as nn
import torch.optim as optim

# ----- Simple ANN -----
class NeurobotBrain(nn.Module):
    def __init__(self):
        super(NeurobotBrain, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input: 1 sensor (distance)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # Output: 3 actions (FORWARD, LEFT, RIGHT)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ----- Example Training Step -----
brain = NeurobotBrain()
optimizer = optim.Adam(brain.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Example input: distance = 15cm
inputs = torch.tensor([[15.0]])  # shape: [batch, features]
target = torch.tensor([1])       # 0=FORWARD,1=LEFT,2=RIGHT

optimizer.zero_grad()
output = brain(inputs.float())
loss = criterion(output, target)
loss.backward()
optimizer.step()

print("ANN output:", output)
