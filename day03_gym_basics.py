
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 创建结果目录
Path("results").mkdir(exist_ok=True)

# 1. 创建环境
env = gym.make("Reacher-v5")

# 2. 固定随机种子
obs, info = env.reset(seed=42)
env.action_space.seed(42)

# 3. 记录实验数据
steps = []
distances = []
rewards = []
reward_distances = []
reward_controls = []

total_reward = 0.0

# 4. 执行一个完整 Episode
for step in range(100):

    # 随机策略
    action = env.action_space.sample()

    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 解析 Observation
    theta1 = np.arctan2(obs[2], obs[0])
    theta2 = np.arctan2(obs[3], obs[1])

    distance = np.linalg.norm(obs[8:10])

    # 获取奖励分量
    r_dist = info["reward_dist"]
    r_ctrl = info["reward_ctrl"]

    # 记录数据
    steps.append(step + 1)
    distances.append(distance)
    rewards.append(reward)
    reward_distances.append(r_dist)
    reward_controls.append(r_ctrl)

    total_reward += reward

    # 每隔 10 步输出一次
    if (step + 1) % 10 == 0:
        print(
            f"Step={step+1}, "
            f"Angle1={np.degrees(theta1):.2f}, "
            f"Angle2={np.degrees(theta2):.2f}, "
            f"Distance={distance:.4f}, "
            f"Reward={reward:.4f}"
        )

    # 5. 判断 Episode 是否结束
    if terminated or truncated:
        print("\nEpisode finished")
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        break

env.close()

print("\n========== 实验结果 ==========")
print("Total steps:", len(steps))
print("Total reward:", round(total_reward, 4))
print("Mean distance:", round(np.mean(distances), 4))
print("Final distance:", round(distances[-1], 4))

# 6. 绘制距离变化
plt.figure(figsize=(8, 5))
plt.plot(steps, distances, label="Distance")
plt.xlabel("Step")
plt.ylabel("Distance (m)")
plt.title("Reacher Distance Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/day03_distance.png", dpi=200)
plt.close()

# 7. 绘制奖励分解
plt.figure(figsize=(8, 5))
plt.plot(steps, rewards, label="Total Reward")
plt.plot(steps, reward_distances, label="Distance Reward")
plt.plot(steps, reward_controls, label="Control Reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reacher Reward Components")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/day03_rewards.png", dpi=200)
plt.close()

print("\n图像已保存到 results 文件夹")
