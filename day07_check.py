
import gymnasium as gym
import mujoco
import torch

print("========== Environment Check ==========")

print("Gymnasium:", gym.__version__)
print("MuJoCo:", mujoco.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 检查 Stable-Baselines3
try:
    import stable_baselines3 as sb3
    print("Stable-Baselines3:", sb3.__version__)
except ImportError:
    print("Stable-Baselines3: NOT INSTALLED")

# 检查 Reacher 环境
env = gym.make("Reacher-v5")
obs, info = env.reset(seed=42)

print("\n========== Reacher Check ==========")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

# 执行一个动作，确保环境能正常运行
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("Reward:", reward)
print("Step successful!")

env.close()

print("\nEnvironment check completed.")
