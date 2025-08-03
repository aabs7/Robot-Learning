from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym


env = make_vec_env('CarRacing-v3', n_envs=8)
model = PPO(
    'CnnPolicy',
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./ppo_car_racing_tensorboard/"
)
model.learn(total_timesteps=1_000_000)

model.save("ppo_car_racing_expert")
env.close()
