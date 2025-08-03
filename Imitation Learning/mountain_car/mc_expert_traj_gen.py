import gymnasium as gym
import numpy as np
import pickle

PUSH_LEFT = 0
PUSH_RIGHT = 2
PUSH_NOOP = 1

def expert_policy(state):
    _, velocity = state
    if velocity > 0:
        return PUSH_RIGHT
    elif velocity < 0:
        return PUSH_LEFT
    else:
        return PUSH_NOOP

if __name__ == "__main__":
    # env = gym.make('MountainCar-v0', render_mode='human')
    env = gym.make('MountainCar-v0')
    expert_trajectory = []

    for _ in range(1000):
        state, _ = env.reset()
        done = False
        while not done:
            action = expert_policy(state)
            print(f"State: {state}, Action: {action}")
            expert_trajectory.append((state, action))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    with open('expert_trajectory.pkl', 'wb') as f:
        pickle.dump(expert_trajectory, f)
