
import gymnasium as gym
import numpy as np
import csv
from pathlib import Path

from stable_baselines3 import PPO

SEEDS = range(20)

OUTPUT_DIR = Path("results/day10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Baseline_w1": "models/ppo_reacher_day08.zip",
    "Experiment_w0.1": "models/day10/ppo_wc_0_1.zip",
    "Experiment_w0.01": "models/day10/ppo_wc_0_01.zip"
}

all_records = []

for name, model_path in MODELS.items():

    print(f"\nEvaluating {name}")

    model = PPO.load(model_path, device="cpu")

    for seed in SEEDS:

        # 统一使用相同的环境配置
        env = gym.make(
            "Reacher-v5",
            reward_dist_weight=1.0,
            reward_control_weight=1.0
        )

        obs, info = env.reset(seed=seed)

        distances = []
        action_costs = []

        try:
            for step in range(50):

                action, _ = model.predict(
                    obs,
                    deterministic=True
                )

                # 记录环境实际接收的动作
                action = np.clip(
                    action,
                    env.action_space.low,
                    env.action_space.high
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                distances.append(
                    float(np.linalg.norm(obs[8:10]))
                )

                action_costs.append(
                    float(np.sum(np.square(action)))
                )

                if terminated or truncated:
                    break

        finally:
            env.close()

        distances = np.asarray(distances)
        action_costs = np.asarray(action_costs)

        # 用相同的标准计算所有模型的分数
        common_return = -float(
            np.sum(distances + action_costs)
        )

        record = {
            "model": name,
            "seed": seed,
            "common_return": common_return,
            "mean_distance": float(distances.mean()),
            "final_distance": float(distances[-1]),
            "min_distance": float(distances.min()),
            "ever_reached": int(
                distances.min() < 0.05
            ),
            "mean_action_norm": float(
                np.mean(np.sqrt(action_costs))
            ),
            "total_action_cost": float(
                np.sum(action_costs)
            )
        }

        all_records.append(record)


# ========== 汇总 ==========

print("\n========== Day 10 Evaluation ==========")

for name in MODELS:

    group = [
        r for r in all_records
        if r["model"] == name
    ]

    def avg(key):
        return float(np.mean([r[key] for r in group]))

    print(f"\nModel: {name}")

    print(
        f"Common Return: "
        f"{avg('common_return'):.4f}"
    )

    print(
        f"Mean Distance: "
        f"{avg('mean_distance'):.4f}"
    )

    print(
        f"Final Distance: "
        f"{avg('final_distance'):.4f}"
    )

    print(
        f"Ever Reached: "
        f"{avg('ever_reached') * 100:.1f}%"
    )

    print(
        f"Mean Action Norm: "
        f"{avg('mean_action_norm'):.4f}"
    )

    print(
        f"Total Action Cost: "
        f"{avg('total_action_cost'):.4f}"
    )


# ========== 保存结果 ==========

csv_path = OUTPUT_DIR / "ablation_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(all_records[0].keys())
    )

    writer.writeheader()
    writer.writerows(all_records)

print("\nSaved:", csv_path)
