
import gymnasium as gym
import numpy as np

env = gym.make("Reacher-v5")

obs, info = env.reset(seed=42)
env.action_space.seed(42)

for step in range(10):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    distance = np.linalg.norm(obs[8:10])

    print(f"\nStep {step+1}")
    print("Action:", action)
    print("Distance:", round(distance, 6))

    print("Reward:", round(reward, 6))
    print("Reward info:", info)

    # 检查奖励分量是否相加得到总奖励
    if "reward_dist" in info and "reward_ctrl" in info:
        total = info["reward_dist"] + info["reward_ctrl"]
        print("Reward sum:", round(total, 6))
        print("Difference:", abs(total - reward))

    if terminated or truncated:
        break

env.close()
