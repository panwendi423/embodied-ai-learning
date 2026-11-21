
import gymnasium as gym
import torch
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_reacher_day08.zip"

# 1. 加载已经训练好的模型
model = PPO.load(MODEL_PATH, device="cpu")

print("========== PPO Model ==========")
print(model.policy)

# 2. 查看主要网络模块
print("\n========== Actor MLP ==========")
print(model.policy.mlp_extractor.policy_net)

print("\n========== Critic MLP ==========")
print(model.policy.mlp_extractor.value_net)

print("\n========== Action Head ==========")
print(model.policy.action_net)

print("\n========== Value Head ==========")
print(model.policy.value_net)

# 3. 统计网络参数
actor_params = (
    sum(p.numel() for p in
        model.policy.mlp_extractor.policy_net.parameters())
    + sum(p.numel() for p in
        model.policy.action_net.parameters())
    + model.policy.log_std.numel()
)

critic_params = (
    sum(p.numel() for p in
        model.policy.mlp_extractor.value_net.parameters())
    + sum(p.numel() for p in
        model.policy.value_net.parameters())
)

total_params = sum(
    p.numel() for p in model.policy.parameters()
)

print("\n========== Parameter Count ==========")
print("Actor parameters:", actor_params)
print("Critic parameters:", critic_params)
print("Total parameters:", total_params)

# 4. 用真实 Observation 测试网络
env = gym.make(
    "Reacher-v5",
    reward_dist_weight=1.0,
    reward_control_weight=1.0
)

obs, info = env.reset(seed=42)

action, _ = model.predict(
    obs,
    deterministic=True
)

print("\n========== Inference ==========")
print("Observation shape:", obs.shape)
print("Action:", action)
print("Action shape:", action.shape)

env.close()
