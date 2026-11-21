
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from pathlib import Path
from stable_baselines3 import PPO

# ---------- 基本配置 ----------

SEED = 42
OUTPUT_DIR = Path("results/day09")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = PPO.load(
    "models/ppo_reacher_day08.zip",
    device="cpu"
)

env = gym.make(
    "Reacher-v5",
    render_mode="rgb_array",
    reward_dist_weight=1.0,
    reward_control_weight=1.0
)

obs, info = env.reset(seed=SEED)

frames = [env.render()]
distances = []
action_norms = []
rewards = []

# ---------- 执行 PPO ----------

for step in range(50):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    distance = float(np.linalg.norm(obs[8:10]))
    action_norm = float(np.linalg.norm(action))

    distances.append(distance)
    action_norms.append(action_norm)
    rewards.append(float(reward))
    frames.append(env.render())

    print(
        f"Step {step+1:02d} | "
        f"Distance={distance:.4f} | "
        f"Action={np.round(action, 4)} | "
        f"Reward={reward:.4f}"
    )

    if terminated or truncated:
        break

fps = env.metadata.get("render_fps", 50)

imageio.mimsave(
    str(OUTPUT_DIR / "ppo_reacher.mp4"),
    frames,
    fps=fps,
    macro_block_size=1
)

env.close()

# ---------- 绘制距离曲线 ----------

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel("Environment Step")
plt.ylabel("Distance (m)")
plt.title("PPO: End-effector Distance")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ppo_distance.png",
    dpi=200
)
plt.close()

# ---------- 绘制动作幅度曲线 ----------

plt.figure(figsize=(8, 4))
plt.plot(action_norms)
plt.xlabel("Environment Step")
plt.ylabel("Action L2 Norm")
plt.title("PPO: Control Action Magnitude")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ppo_action_norm.png",
    dpi=200
)
plt.close()

print("\n========== Summary ==========")
print("Total reward:", round(sum(rewards), 4))
print("Mean distance:", round(np.mean(distances), 4))
print("Final distance:", round(distances[-1], 4))
print("Min distance:", round(min(distances), 4))
print("Mean action norm:", round(np.mean(action_norms), 4))
print("Video and charts saved.")
