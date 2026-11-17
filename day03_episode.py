
import gymnasium as gym
import numpy as np

env = gym.make("Reacher-v5")

obs, info = env.reset(seed=42)

total_reward = 0.0

for step in range(100):
    action = np.zeros(2, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    print(
        f"Step={step+1}, "
        f"Reward={reward:.4f}, "
        f"Terminated={terminated}, "
        f"Truncated={truncated}"
    )

    if terminated or truncated:
        print("\nEpisode 结束")
        print("实际步数:", step + 1)
        print("累计奖励:", total_reward)
        break

env.close()
