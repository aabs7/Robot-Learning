import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from utils import PolicyNetwork, ValueNetwork, collect_trajectory, evaluate_policy


def train_policy_gradient_with_advantage(env, batch_size=5000, num_epochs=1000, gamma=0.99, lr=1e-3):
    print(f"Training in {env.spec.id}")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(obs_dim, action_dim)
    value_net = ValueNetwork(obs_dim)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        trajectory = collect_trajectory(env, policy_net, batch_size, gamma)
        value_pred = value_net(trajectory['observations'])
        # Compute advantages
        advantages = trajectory['rewards'] - value_pred.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy update
        log_probs = policy_net.get_log_probs(trajectory['observations'], trajectory['actions'])
        policy_loss = -(log_probs * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Value function fit
        for _ in range(80):
            value_pred = value_net(trajectory['observations']) # replacing variable so no gradient tracking from above
            value_loss = ((value_pred - trajectory['rewards']) ** 2).mean()
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        print(f"Epoch {epoch}: Policy Loss = {policy_loss.item():.3f}, Value Loss = {value_loss.item():.3f}")
    torch.save(policy_net.state_dict(), f'{env.spec.id}_advantage_policy_net.pth')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--just_evaluate', action='store_true', default=False, help='Just evaluate the policy without training')
    args = parser.parse_args()

    if not args.just_evaluate:
        env = gym.make(args.env_name)
        print("Environment created:", env.spec.id)
        print("Training ...")
        train_policy_gradient_with_advantage(env, args.batch_size, args.num_epochs, args.gamma, args.lr)
        env.close()

    print("Evaluating ...")
    env = gym.make(args.env_name, render_mode='human')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(obs_dim, action_dim)
    policy_net.load_state_dict(torch.load(f'{env.spec.id}_advantage_policy_net.pth'))
    evaluate_policy(env, policy_net)
