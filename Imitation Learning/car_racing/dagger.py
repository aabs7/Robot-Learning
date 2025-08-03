import pickle
import gymnasium as gym
import torch
import numpy as np
from cr_behavior_cloning_train import BCCNN, train
from stable_baselines3 import PPO

DAGGER_NUM_ITERATIONS = 5
DAGGER_EPISODES_PER_ITER = 5

if __name__ == "__main__":
    # load the dataset to append to
    expert_trajectory = []
    with open('expert_trajectory.pkl', 'rb') as f:
        expert_trajectory = pickle.load(f)


    # Load the BC policy model
    model_bc = BCCNN()
    model_bc.load_state_dict(torch.load('behavior_cloning_model.pth'))
    model_bc.eval()

    # Load the PPO expert model
    model_ppo = PPO.load("ppo_car_racing_expert")


    for iteration in range(DAGGER_NUM_ITERATIONS):
        print(f"Starting DAGGER iteration {iteration + 1}/{DAGGER_NUM_ITERATIONS}")
        dagger_data = []
        env = gym.make("CarRacing-v3", render_mode="human")
        for episode in range(DAGGER_EPISODES_PER_ITER):
            print(f"Episode {episode + 1}/{DAGGER_EPISODES_PER_ITER}")
            state, _ = env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_tensor = state_tensor.permute(0, 3, 1, 2) / 255.0
                with torch.no_grad():
                    action_pred = model_bc(state_tensor).numpy().flatten()
                action_expert, _ = model_ppo.predict(state)

                dagger_data.append((state, action_expert))
                state, _, terminated, truncated, _ = env.step(action_pred)
                done = terminated or truncated
        expert_trajectory.extend(dagger_data)

        env.close()
        # Continue training the BC model with the new data
        print("Retraining BC model with the new data")
        train(expert_trajectory, model=model_bc, num_epochs=20)
