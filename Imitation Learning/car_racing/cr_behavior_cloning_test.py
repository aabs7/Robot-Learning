import gymnasium as gym
import torch
from cr_behavior_cloning_train import BCCNN

if __name__ == "__main__":
    # Load the trained model
    model = BCCNN()
    model.load_state_dict(torch.load('behavior_cloning_model.pth'))
    model.eval()

    env = gym.make("CarRacing-v3", render_mode="human")

    for episode in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
            state_tensor = state_tensor / 255.0
            with torch.no_grad():
                logits = model(state_tensor)
                action = logits.numpy().flatten()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
