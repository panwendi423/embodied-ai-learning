
import gymnasium as gym

# 1. 创建机械臂仿真环境
env = gym.make("Reacher-v5")

# 2. 初始化机器人状态
obs, info = env.reset(seed=42)


import numpy as np

print("\n机器人初始状态：")

print("关节1 cos:", obs[0])
print("关节2 cos:", obs[1])

print("关节1 sin:", obs[2])
print("关节2 sin:", obs[3])

print("目标位置:", obs[4:6])

print("关节角速度:", obs[6:8])

print("末端与目标的相对位置:", obs[8:10])

distance = np.linalg.norm(obs[8:10])
print("末端与目标距离:", distance)


print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

# 3. 执行10次随机动作
for step in range(10):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step}: "
        f"Action={action}, "
        f"Reward={reward:.4f}"
    )

    if terminated or truncated:
        break

# 4. 关闭仿真
env.close()
