import gymnasium as gym
import torch
from mc_behavior_cloning_train import BCModel

if __name__ == "__main__":
    # Load the trained model
    model = BCModel()
    model.load_state_dict(torch.load('bc_model.pth'))
    model.eval()

    env = gym.make('MountainCar-v0', render_mode='human')

    for episode in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                print(f"Current state: {state}")
                logits = model(state_tensor)
                print(f"Logits: {logits}")
                action = torch.argmax(logits, dim=1).item()
                print(f"Predicted action: {action}")

            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
