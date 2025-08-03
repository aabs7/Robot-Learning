import gymnasium as gym
import numpy as np
import pygame
import pickle
import time


def get_keyboard_action():
    steer = 0.0
    gas = 0.0
    brake = 0.0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        steer = -0.5
    elif keys[pygame.K_RIGHT]:
        steer = 0.5

    if keys[pygame.K_UP]:
        gas = 0.2

    if keys[pygame.K_DOWN]:
        brake = 0.1

    return np.array([steer, gas, brake], dtype=np.float32)

import matplotlib.pyplot as plt
if __name__ == "__main__":

    pygame.init()
    # Initialize environment with visual output
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset()

    expert_trajectory = []
    episodes = 5

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\nStarting episode {ep + 1}")
        total_reward = 0
        while not done:
            # Pump pygame event queue to get keyboard input
            pygame.event.pump()
            action = get_keyboard_action()
            # Save the current frame and action
            expert_trajectory.append((state, action))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # time.sleep(0.01)  # Slow down for human control
            total_reward += reward
        print(f"Total reward: {total_reward}")

    # Save the collected states and actions
    with open('expert_trajectory.pkl', 'wb') as f:
        pickle.dump(expert_trajectory, f)
