
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np


# Simple NN for behavior cloning
# 96 x 96 x 3 image
class BCCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output: steer, gas, brake
        )

    def forward(self, x):
        return self.net(x)

def train(expert_trajectory, model=None, num_epochs=200):
    states = torch.tensor(np.array([s for s, a in expert_trajectory], dtype=np.float32))
    actions = torch.tensor(np.array([a for s, a in expert_trajectory], dtype=np.float32))

    states = states / 255.0 # Normalize
    states = states.permute(0, 3, 1, 2)  # Change to (N, C, H, W)

    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    if model is None:
        model = BCCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = num_epochs

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            preds = model(batch_states)
            loss = criterion(preds, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    torch.save(model.state_dict(), 'behavior_cloning_model.pth')

if __name__ == "__main__":
    with open('expert_trajectory.pkl', 'rb') as f:
        expert_trajectory = pickle.load(f)

    train(expert_trajectory)
    print("Training complete!!")
