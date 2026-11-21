
import gymnasium as gym
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------- 1. 基本配置 ----------

SEED = 42

# 先用 2048 检查程序，再改为 20000
TOTAL_TIMESTEPS = 20000

MODEL_DIR = Path("models")
LOG_DIR = Path("results/day08")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)

# ---------- 2. 创建环境 ----------

env = gym.make(
    "Reacher-v5",
    reward_dist_weight=1.0,
    reward_control_weight=1.0
)

# Monitor 记录每个 Episode 的长度与累计奖励
env = Monitor(
    env,
    filename=str(LOG_DIR / "monitor.csv")
)

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Device: CPU")
print("Training timesteps:", TOTAL_TIMESTEPS)

# ---------- 3. 创建 PPO 模型 ----------

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    device="cpu",
    verbose=1,
    seed=SEED,
    tensorboard_log=str(LOG_DIR / "tensorboard")
)

print("\nPPO model created successfully.")

# ---------- 4. 定期保存检查点 ----------

checkpoint_callback = CheckpointCallback(
    save_freq=10240,
    save_path=str(MODEL_DIR),
    name_prefix="ppo_reacher_checkpoint"
)

# ---------- 5. 开始训练 ----------

print("\nTraining started...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=False
)

# ---------- 6. 保存最终模型 ----------

model_path = MODEL_DIR / "ppo_reacher_day08"

model.save(str(model_path))

print("\nTraining completed.")
print("Model saved:", str(model_path) + ".zip")

env.close()
