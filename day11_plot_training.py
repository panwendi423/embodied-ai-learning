
from pathlib import Path
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import (
    event_accumulator
)

# ========== 1. 配置 ==========

OUTPUT_DIR = Path("results/day11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ROOTS = {
    "Baseline_w1":
        Path("results/day08/tensorboard"),

    "Experiment_w0.1":
        Path("results/day10/tensorboard"),

    "Experiment_w0.01":
        Path("results/day10/tensorboard")
}

# Day 10 的运行目录前缀
RUN_PREFIXES = {
    "Baseline_w1": "PPO",
    "Experiment_w0.1": "wc_0_1",
    "Experiment_w0.01": "wc_0_01"
}

METRICS = [
    "rollout/ep_rew_mean",
    "train/explained_variance",
    "train/value_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/std"
]


# ========== 2. 读取单个运行目录 ==========

def read_run(run_dir):

    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.SCALARS: 0
        }
    )

    accumulator.Reload()

    tags = accumulator.Tags().get("scalars", [])

    data = {}

    for metric in METRICS:

        if metric not in tags:
            continue

        events = accumulator.Scalars(metric)

        # 按实际环境步数绘制
        steps = [e.step for e in events]
        values = [e.value for e in events]

        data[metric] = (steps, values)

    return data


# ========== 3. 自动寻找实验目录 ==========

def select_run(root, prefix):

    if not root.exists():
        raise FileNotFoundError(
            f"Log folder not found: {root}"
        )

    # 查找带有 TensorBoard event 文件的目录
    event_files = list(
        root.rglob("events.out.tfevents.*")
    )

    candidates = {}

    for file in event_files:

        directory = file.parent

        # 通过运行目录前缀区分三组实验
        if directory.name.startswith(prefix + "_"):
            candidates[directory] = True

    if not candidates:
        raise FileNotFoundError(
            f"No TensorBoard run for {prefix} in {root}"
        )

    # Day 8 曾经进行过 smoke test：
    # 优先选择记录步数最多的完整训练
    best_run = None
    best_steps = -1

    for directory in candidates:

        data = read_run(directory)

        all_steps = [
            max(steps)
            for steps, values in data.values()
            if steps
        ]

        if not all_steps:
            continue

        max_step = max(all_steps)

        if max_step > best_steps:
            best_steps = max_step
            best_run = directory

    if best_run is None:
        raise RuntimeError(
            f"No scalar data found for {prefix}"
        )

    print(
        f"Selected {prefix}: "
        f"{best_run} ({best_steps} steps)"
    )

    return best_run


# ========== 4. 读取全部实验 ==========

all_data = {}

for name, root in RUN_ROOTS.items():

    run_dir = select_run(
        root,
        RUN_PREFIXES[name]
    )

    all_data[name] = read_run(run_dir)


# ========== 5. 每个指标单独生成图表 ==========

for metric in METRICS:

    plt.figure(figsize=(9, 5))

    plotted = False

    for name, run_data in all_data.items():

        if metric not in run_data:
            print(f"Missing {metric} for {name}")
            continue

        steps, values = run_data[metric]

        plt.plot(
            steps,
            values,
            label=name,
            linewidth=1.8
        )

        plotted = True

    if not plotted:
        plt.close()
        continue

    plt.xlabel("Training Timesteps")
    plt.ylabel(metric.split("/")[-1])
    plt.title(metric)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    filename = metric.replace("/", "_") + ".png"

    save_path = OUTPUT_DIR / filename

    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)


print("\nDay 11 training plots completed.")
