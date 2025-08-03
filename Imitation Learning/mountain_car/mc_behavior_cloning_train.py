import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np


# Simple NN for behavior cloning
class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Load the expert trajectory
    with open('expert_trajectory.pkl', 'rb') as f:
        expert_trajectory = pickle.load(f)

    states = np.array([s for s, a in expert_trajectory], dtype=np.float32)
    actions = np.array([a for s, a in expert_trajectory], dtype=np.int64)

    model = BCModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000

    for epoch in range(num_epochs):
        logits = model(torch.tensor(states))
        loss = criterion(logits, torch.tensor(actions))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    print("Training complete.")

    with open('bc_model.pth', 'wb') as f:
        torch.save(model.state_dict(), f)
