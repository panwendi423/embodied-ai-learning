
import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

# 导入 Day 4 编写的策略
from day04_control import policies

# 创建输出目录
output_dir = Path("results/videos")
output_dir.mkdir(parents=True, exist_ok=True)


def record_policy(name, policy, seed=42):

    # 使用可返回图像的渲染模式
    env = gym.make(
        "Reacher-v5",
        render_mode="rgb_array"
    )

    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    frames = []
    total_reward = 0.0
    distances = []

    # 保存初始画面
    frames.append(env.render())

    for step in range(50):

        # 根据对应策略生成动作
        action = policy(obs, step, rng)

        # 执行机器人动作
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        distances.append(np.linalg.norm(obs[8:10]))

        # 保存当前机器人画面
        frames.append(env.render())

        if terminated or truncated:
            break

    # 视频输出路径
    video_path = output_dir / f"reacher_{name.lower()}.mp4"

    fps = env.metadata.get("render_fps", 50)

    imageio.mimsave(
        str(video_path),
        frames,
        fps=fps,
        macro_block_size=1
    )

    print(f"\nPolicy: {name}")
    print(f"Video: {video_path}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Mean distance: {np.mean(distances):.4f}")

    env.close()


if __name__ == "__main__":

    for name, policy in policies.items():
        record_policy(name, policy)
