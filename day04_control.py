
import gymnasium as gym
import numpy as np

# ---------- 三种控制策略 ----------

def random_policy(obs, step, rng):
    return rng.uniform(
        -1, 1, size=2
    ).astype(np.float32)


def zero_policy(obs, step, rng):
    return np.zeros(
        2, dtype=np.float32
    )


def periodic_policy(obs, step, rng):
    return np.array([
        0.5 * np.sin(step * 0.1),
        0.5 * np.cos(step * 0.1)
    ], dtype=np.float32)


policies = {
    "Random": random_policy,
    "Zero": zero_policy,
    "Periodic": periodic_policy
}


# ---------- 运行一个 Episode ----------

def evaluate_policy(policy, seed=42):
    env = gym.make("Reacher-v5")
    obs, info = env.reset(seed=seed)

    rng = np.random.default_rng(seed)

    total_reward = 0.0
    distances = []

    for step in range(50):
        action = policy(obs, step, rng)

        obs, reward, terminated, truncated, info = env.step(action)

        distance = np.linalg.norm(obs[8:10])

        total_reward += reward
        distances.append(distance)

        if terminated or truncated:
            break

    env.close()

    return {
        "total_reward": total_reward,
        "mean_distance": np.mean(distances),
        "final_distance": distances[-1]
    }


# ---------- 主程序 ----------

if __name__ == "__main__":

    for name, policy in policies.items():

        result = evaluate_policy(policy)

        print(f"\nPolicy: {name}")
        print(
            f"Total reward: "
            f"{result['total_reward']:.4f}"
        )
        print(
            f"Mean distance: "
            f"{result['mean_distance']:.4f}"
        )
        print(
            f"Final distance: "
            f"{result['final_distance']:.4f}"
        )
