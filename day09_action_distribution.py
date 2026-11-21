
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

model = PPO.load(
    "models/ppo_reacher_day08.zip",
    device="cpu"
)

env = gym.make(
    "Reacher-v5",
    reward_dist_weight=1.0,
    reward_control_weight=1.0
)

obs, info = env.reset(seed=42)

# 将 Observation 转为网络需要的 Tensor
obs_tensor, _ = model.policy.obs_to_tensor(obs)

with torch.no_grad():
    distribution = model.policy.get_distribution(obs_tensor)

    mean = distribution.distribution.mean
    std = distribution.distribution.stddev

print("========== Gaussian Distribution ==========")
print("Mean:", mean.cpu().numpy().flatten())
print("Std:", std.cpu().numpy().flatten())

print("\n========== Deterministic Action ==========")
action, _ = model.predict(obs, deterministic=True)
print(action)

print("\n========== Stochastic Actions ==========")

actions = []

for i in range(10):
    action, _ = model.predict(
        obs,
        deterministic=False
    )

    actions.append(action)
    print(f"Sample {i + 1}: {action}")

actions = np.array(actions)

print("\nSample action mean:", actions.mean(axis=0))
print("Sample action std:", actions.std(axis=0))

env.close()
