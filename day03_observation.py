
import gymnasium as gym
import numpy as np

env = gym.make("Reacher-v5")
obs, info = env.reset(seed=42)

# 计算关节角度
theta1 = np.arctan2(obs[2], obs[0])
theta2 = np.arctan2(obs[3], obs[1])

# 读取目标与关节速度
target = obs[4:6]
velocities = obs[6:8]

# 计算末端相对目标的距离
relative_pos = obs[8:10]
distance = np.linalg.norm(relative_pos)

print("关节1角度:", np.degrees(theta1), "度")
print("关节2角度:", np.degrees(theta2), "度")

print("目标位置:", target)
print("角速度:", velocities)

print("相对位置:", relative_pos)
print("末端与目标距离:", distance)

env.close()
