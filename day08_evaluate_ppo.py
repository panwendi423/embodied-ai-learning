
import gymnasium as gym
import numpy as np
import csv
from pathlib import Path

from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_reacher_day08.zip"

SEEDS = list(range(20))
DISTANCE_THRESHOLD = 0.05

OUTPUT_DIR = Path("results/day08")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. 加载 PPO 模型 ----------

model = PPO.load(
    MODEL_PATH,
    device="cpu"
)

print("PPO model loaded successfully.")

# ---------- 2. 多随机种子评估 ----------

records = []

for seed in SEEDS:

    env = gym.make(
        "Reacher-v5",
        reward_dist_weight=1.0,
        reward_control_weight=1.0
    )

    obs, info = env.reset(seed=seed)

    rewards = []
    distances = []
    control_rewards = []

    try:
        for step in range(50):

            # 测试阶段使用确定性动作
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            rewards.append(float(reward))
            distances.append(
                float(np.linalg.norm(obs[8:10]))
            )
            control_rewards.append(
                float(info["reward_ctrl"])
            )

            if terminated or truncated:
                break

    finally:
        env.close()

    result = {
        "seed": seed,
        "total_reward": float(np.sum(rewards)),
        "mean_distance": float(np.mean(distances)),
        "final_distance": float(distances[-1]),
        "min_distance": float(np.min(distances)),
        "total_control_reward": float(
            np.sum(control_rewards)
        ),
        "ever_reached": int(
            np.min(distances) < DISTANCE_THRESHOLD
        ),
        "steps": len(rewards)
    }

    records.append(result)

    print(
        f"Seed {seed:02d} | "
        f"Return={result['total_reward']:.4f} | "
        f"MeanDist={result['mean_distance']:.4f}"
    )

# ---------- 3. 计算汇总指标 ----------

returns = np.array([
    r["total_reward"] for r in records
])

mean_distances = np.array([
    r["mean_distance"] for r in records
])

final_distances = np.array([
    r["final_distance"] for r in records
])

ever_reached = np.array([
    r["ever_reached"] for r in records
])

print("\n========== PPO Evaluation ==========")

print(f"Mean Return: {returns.mean():.4f}")
print(f"Std Return: {returns.std(ddof=1):.4f}")
print(
    f"Mean Distance: "
    f"{mean_distances.mean():.4f}"
)
print(
    f"Mean Final Distance: "
    f"{final_distances.mean():.4f}"
)
print(
    f"Ever Reached (<0.05m): "
    f"{ever_reached.mean() * 100:.1f}%"
)

# ---------- 4. 保存 CSV ----------

csv_path = OUTPUT_DIR / "ppo_evaluation.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print("\nResults saved:", csv_path)
