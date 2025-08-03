import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load trained PPO expert
model = PPO.load("ppo_car_racing_expert")

# Set up environment with human rendering
env = gym.make("CarRacing-v3", render_mode="human")

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    # Expert predicts action
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Expert total reward:", total_reward)

env.close()
