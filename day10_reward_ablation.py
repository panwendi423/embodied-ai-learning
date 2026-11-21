
import gymnasium as gym
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ========== 1. 实验配置 ==========

SEED = 42
TOTAL_TIMESTEPS = 20480

CONTROL_WEIGHTS = [0.1, 0.01]

MODEL_DIR = Path("models/day10")
LOG_DIR = Path("results/day10")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)


# ========== 2. 逐组训练 ==========

for weight in CONTROL_WEIGHTS:

    tag = f"wc_{weight:g}".replace(".", "_")

    print("\n" + "=" * 45)
    print(f"Experiment: {tag}")
    print(f"Control Weight: {weight}")
    print(f"Training Steps: {TOTAL_TIMESTEPS}")
    print("=" * 45)

    # 保持距离权重为 1
    env = gym.make(
        "Reacher-v5",
        reward_dist_weight=1.0,
        reward_control_weight=weight
    )

    # 每个实验独立保存日志
    env = Monitor(
        env,
        filename=str(LOG_DIR / f"{tag}_monitor.csv")
    )

    # 每组均从头初始化，不加载旧模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        seed=SEED,
        device="cpu",
        verbose=1,
        tensorboard_log=str(LOG_DIR / "tensorboard")
    )

    # 开始训练
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=tag
    )

    # 各组模型使用不同文件名
    save_path = MODEL_DIR / f"ppo_{tag}"

    model.save(str(save_path))

    print(f"\nSaved: {save_path}.zip")

    env.close()

print("\nAll ablation experiments completed.")
