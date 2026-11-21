
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

model = PPO.load(
    "models/ppo_reacher_day08.zip",
    device="cpu"
)

env = gym.make(
    "Reacher-v5",
    reward_dist_weight=1.0,
    reward_control_weight=1.0
)

print("========== Test A: Different Seeds ==========")

for seed in [0, 1, 2, 3, 4]:
    obs, info = env.reset(seed=seed)

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    print(
        f"Seed={seed} | "
        f"Target={np.round(obs[4:6], 4)} | "
        f"Action={np.round(action, 4)}"
    )

print("\n========== Test B: Target Perturbation ==========")

obs, info = env.reset(seed=42)

base_action, _ = model.predict(
    obs,
    deterministic=True
)

print("Original:", np.round(base_action, 5))

# 在固定机械臂状态下，对目标 x 坐标做小幅扰动
# obs[4] 是目标 x
# obs[8] 是末端与目标的相对 x，具体符号关系需保持一致
for delta in [-0.02, -0.01, 0.01, 0.02]:

    modified_obs = obs.copy()

    modified_obs[4] += delta
    modified_obs[8] += delta

    action, _ = model.predict(
        modified_obs,
        deterministic=True
    )

    difference = np.linalg.norm(
        action - base_action
    )

    print(
        f"Target x shift={delta:+.3f} | "
        f"Action={np.round(action, 5)} | "
        f"Action change={difference:.6f}"
    )

env.close()
