import torch
import torch.nn as nn
import numpy as np


# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, observations):
        logits = self.net(observations)
        return torch.distributions.Categorical(logits=logits)

    def get_log_probs(self, observations, actions):
        logits = self.forward(observations)
        return logits.log_prob(actions)

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, observations):
        return self.net(observations)

# discounted rewards to go
def rewards_to_go(rewards, gamma=0.99):
    result = []
    cum = 0
    for r in reversed(rewards):
        cum = r + gamma * cum
        result.insert(0, cum)
    return result

# Trajectory Collection
def collect_trajectory(env, policy_net, batch_size=5000, gamma=0.99):
    observations, actions, rewards = [], [], []
    obs, _ = env.reset()
    done = False
    episode_rewards = []
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action_logits = policy_net(obs_tensor)
        action = action_logits.sample().item()

        observations.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_rewards.append(reward)
        if done:
            rewards.extend(rewards_to_go(episode_rewards, gamma))
            episode_rewards = []
            done = False
            obs, _ = env.reset()
            if len(observations) >= batch_size:
                break

    return {
        'observations': torch.tensor(np.array(observations), dtype=torch.float32),
        'actions': torch.tensor(np.array(actions), dtype=torch.int64),
        'rewards': torch.tensor(np.array(rewards), dtype=torch.float32),
    }

# Evaluate policy_net in the Gym environment
def evaluate_policy(env, policy_net, episodes=5):
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy_net(obs_tensor).sample().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f'Episode {i + 1} Reward: {total_reward}')
