
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from day04_control import policies

OUTPUT_DIR = Path("results/day06")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(20))
MAX_STEPS = 50

# 仅用于判断一次 Episode 是否曾接近目标
DISTANCE_THRESHOLD = 0.05


def evaluate_one_episode(policy, seed):
    env = gym.make("Reacher-v5")
    obs, info = env.reset(seed=seed)

    # 独立、可复现的策略随机数生成器
    rng = np.random.default_rng(seed)

    rewards = []
    distances = []
    control_rewards = []

    try:
        for step in range(MAX_STEPS):
            action = policy(obs, step, rng)

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

    return {
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


# ---------- 运行所有实验 ----------

records = []

for policy_name, policy_fn in policies.items():

    print(f"\nEvaluating {policy_name}...")

    for seed in SEEDS:
        result = evaluate_one_episode(
            policy_fn, seed
        )

        result["policy"] = policy_name
        records.append(result)

        print(
            f"Seed {seed:02d} | "
            f"Return={result['total_reward']:.3f} | "
            f"MeanDist={result['mean_distance']:.3f}"
        )


# ---------- 保存完整结果 ----------

csv_path = OUTPUT_DIR / "evaluation_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\nCSV saved: {csv_path}")


# ---------- 统计汇总 ----------

summary = {}

print("\n========== Evaluation Summary ==========")

for policy_name in policies.keys():
    policy_records = [
        r for r in records
        if r["policy"] == policy_name
    ]

    returns = np.array([
        r["total_reward"] for r in policy_records
    ])

    mean_distances = np.array([
        r["mean_distance"] for r in policy_records
    ])

    final_distances = np.array([
        r["final_distance"] for r in policy_records
    ])

    ever_reached = np.array([
        r["ever_reached"] for r in policy_records
    ])

    summary[policy_name] = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns, ddof=1),
        "mean_distance": np.mean(mean_distances),
        "mean_final_distance": np.mean(final_distances),
        "reach_rate": np.mean(ever_reached)
    }

    s = summary[policy_name]

    print(f"\nPolicy: {policy_name}")
    print(f"Mean Return: {s['mean_return']:.4f}")
    print(f"Std Return: {s['std_return']:.4f}")
    print(f"Mean Distance: {s['mean_distance']:.4f}")
    print(
        f"Mean Final Distance: "
        f"{s['mean_final_distance']:.4f}"
    )
    print(
        f"Ever Reached (<0.05m): "
        f"{s['reach_rate'] * 100:.1f}%"
    )


# ---------- 图 1：平均奖励与标准差 ----------

names = list(summary.keys())

means = [
    summary[n]["mean_return"] for n in names
]

stds = [
    summary[n]["std_return"] for n in names
]

plt.figure(figsize=(8, 5))
plt.bar(names, means, yerr=stds, capsize=5)
plt.ylabel("Episode Return")
plt.title("Policy Evaluation: Mean and Std")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "mean_reward.png",
    dpi=200
)
plt.close()


# ---------- 图 2：累计奖励箱线图 ----------

reward_groups = [
    [
        r["total_reward"]
        for r in records if r["policy"] == name
    ]
    for name in names
]

plt.figure(figsize=(8, 5))
plt.boxplot(reward_groups, tick_labels=names)
plt.ylabel("Episode Return")
plt.title("Return Distribution Across 20 Seeds")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "reward_boxplot.png",
    dpi=200
)
plt.close()

print("\nAll experiments completed.")
