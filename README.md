
# Embodied AI Learning: MuJoCo & Reacher

## 1. Project Overview

This repository documents my hands-on learning
of robot simulation and reinforcement learning
using MuJoCo and Gymnasium.

## 2. Environment

- Python 3.11
- MuJoCo
- Gymnasium
- Reacher-v5
- NumPy
- Matplotlib

## 3. Experiments

### Day 1–2: MuJoCo Basics
- Joint torque control
- Damping and joint dynamics
- Angular position and velocity analysis

### Day 3: Gymnasium Interaction
- Observation and action spaces
- Reward decomposition
- Episode termination and truncation

### Day 4–5: Baseline Policies
- Random Policy
- Zero Policy
- Periodic Policy
- Robot motion visualization

### Day 6: Multi-seed Evaluation
- 20 random seeds per policy
- Mean episode return
- Standard deviation
- End-effector distance
- Target proximity evaluation

## 4. Evaluation Results

| Policy | Mean Return | Std | Mean Distance |
|---|---:|---:|---:|
| Random | -43.3184 | 4.0460 | 0.2023 |
| Zero | -11.5146 | 4.9771 | 0.2303 |
| Periodic | -21.0289 | 1.6945 | 0.1706 |

## 5. Key Findings

- Zero Policy achieves the highest episode return
  due to its zero control penalty.
- Periodic Policy achieves the lowest mean
  end-effector distance in this evaluation.
- Episode return alone cannot fully describe
  robot task performance.
- All evaluated baselines are open-loop policies.

## 6. Next Steps

- Implement and train a PPO agent.
- Compare PPO with existing baseline policies.
- Evaluate generalization across random seeds.
```

这些数据直接来自你的 Day 6 真实实验，不是示例数据。

可以进一步在 README 中加入 Day 2、Day 3、Day 6 的结果图，并为三种策略的视频添加链接。

## 2. GitHub 上传流程

先检查 Git：

```powershell
git --version
```

若已安装，在项目目录中执行：

```powershell
git init
git add .
git commit -m "Initial commit: MuJoCo and Gymnasium baselines"
```

然后在 GitHub 创建一个空仓库，例如：

`embodied-ai-learning`

将 GitHub 给出的仓库地址代入：

```powershell
git branch -M main
git remote add origin https://github.com/你的用户名/embodied-ai-learning.git
git push -u origin main
```

如果 Git 提示没有设置作者身份，按提示配置 `user.name` 和 `user.email` 即可。

上传前先执行 `git status` 检查文件列表。视频和大文件可以根据实际体积选择是否提交到仓库。

---

# 第三部分：强化学习必学理论（第 3 小时）

这部分是今天最重要的理论内容。

你已经能够写出机器人控制策略，但现在还没有训练过一个真正能学习的策略。

要理解为什么 PPO 能够学习，需要先掌握四个概念。

## 1. 马尔可夫决策过程（MDP）

强化学习通常将交互问题描述为：

\[
\boxed{\mathcal{M}=(S,A,P,R,\gamma)}
\]

| 符号 | 含义 | 在 Reacher 中的例子 |
|---|---|---|
| \(S\) | 状态空间 | 关节角度、速度、目标位置 |
| \(A\) | 动作空间 | 两个关节的控制输入 |
| \(P\) | 状态转移机制 | MuJoCo 动力学仿真 |
| \(R\) | 奖励函数 | 距离惩罚与控制惩罚 |
| \(\gamma\) | 折扣因子 | 决定未来奖励的权重 |

严格来说，Gymnasium 返回的是 Observation；在 Reacher 这类低维状态环境中，可以将其作为策略的状态输入。对于图像观测等情况，Observation 并不一定包含完整环境状态。

你现在需要建立这个关系：

<box border radius="lg" padding={3} gap={2}>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        <title size="lg" color="default">\(s_t\)</title>
        <text color="secondary" size="xs">当前状态</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        <title size="lg" color="default">\(a_t\)</title>
        <text color="secondary" size="xs">采取动作</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center"><icon name="arrow-down"/></row>
  <box background="rgba(22,119,94,.08)" radius="md" padding={3} align="center" gap={1}>
    **环境状态转移 \(P(s_{t+1}\mid s_t,a_t)\)**
    <text color="secondary" size="xs">MuJoCo 根据动力学推进机器人状态</text>
  </box>
  <row justify="center"><icon name="arrow-down"/></row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        <title size="lg" color="default">\(s_{t+1}\)</title>
        <text color="secondary" size="xs">下一个状态</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        <title size="lg" color="default">\(r_t\)</title>
        <text color="secondary" size="xs">本次奖励</text>
      </box>
    </grid-item>
  </grid>
</box>

## 2. Policy：策略

你 Day 4 写的三种 Policy 都是预先设计好的规则。

例如：

```python
def zero_policy(obs, step, rng):
    return np.zeros(2, dtype=np.float32)
```

虽然函数接收 `obs`，但是并没有真正使用它。

后续 PPO 的策略则可以写成：

\[
a_t\sim\pi_\theta(\cdot\mid s_t)
\]

这里的 \(\theta\) 就是神经网络参数。

模型根据当前状态输出动作分布，训练过程会不断更新这些参数。

这是从**手工控制**转向**学习控制**的关键变化。

## 3. Return：累计折扣回报

强化学习关注的不只是眼前的奖励，还包括未来奖励。

\[
\boxed{G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}}
\]

其中 \(\gamma\in[0,1]\) 通常称为折扣因子。

例如，假设未来连续三步的奖励分别是 1、2、3：

{@body const [gamma,setGamma]=DIL.useState(0.9)}
{@body const total=1+gamma*2+gamma*gamma*3}
<box border radius="lg" padding={3} gap={3}>
  <row align="center" justify="between">
    **折扣因子实验**
    <text color="secondary" weight="medium" tabularNums>\(\gamma\) = {gamma.toFixed(2)}</text>
  </row>
  <slider min={0} max={1} step={0.05} value={gamma} onChange={setGamma}/>
  <grid columns={3} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} align="center" gap={1}>
        <text size="xs" color="secondary">立即奖励</text>
        <title size="lg">1.00</title>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} align="center" gap={1}>
        <text size="xs" color="secondary">下一步贡献</text>
        <title size="lg" tabularNums>{(2*gamma).toFixed(2)}</title>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} align="center" gap={1}>
        <text size="xs" color="secondary">下两步贡献</text>
        <title size="lg" tabularNums>{(3*gamma*gamma).toFixed(2)}</title>
      </box>
    </grid-item>
  </grid>
  <divider color="subtle"/>
  <row align="center" justify="between">
    **累计折扣回报**
    <title size="xl" tabularNums>{total.toFixed(3)}</title>
  </row>
  <text color="secondary" size="xs">调整滑块，观察未来奖励的权重如何变化。</text>
</box>

你 Day 6 计算的 Episode Return 是未折扣的累计奖励，相当于对所有步骤的奖励直接求和。

而 PPO 训练通常会使用小于 1 的折扣因子。

## 4. Value Function：价值函数

价值函数用于估计从当前状态出发，未来可能获得多少累计折扣奖励。

\[
\boxed{V^\pi(s)=\mathbb{E}_{\pi}[G_t\mid s_t=s]}
\]

你可以这样理解：

- Reward：当前一步得到了多少奖励。
- Return：从现在开始实际累计获得多少奖励。
- Value：根据当前策略，预测从当前状态开始能够获得多少奖励。

对于 PPO，价值函数非常重要，因为它会参与判断一个动作是否比预期更好。

---

# 第四部分：PPO 算法入门（第 4 小时）

PPO 全称为 Proximal Policy Optimization，是一种常见的深度强化学习算法。

你今天不需要推导完整的 PPO 损失函数，也不需要开始正式训练，只需要理解它的三个关键组件。

## 1. Actor：动作决策网络

Actor 是负责生成动作的策略网络。

输入为 Reacher 的 10 维 Observation，输出与两个关节动作相关的分布参数。

<box border radius="lg" padding={3} gap={2}>
  <box background="surface-secondary" radius="md" padding={3} align="center">
    **10 维 Observation**
  </box>
  <row justify="center"><icon name="arrow-down"/></row>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **Actor 神经网络**
    <text color="secondary" size="xs">根据状态计算动作分布</text>
  </box>
  <row justify="center"><icon name="arrow-down"/></row>
  <box background="rgba(22,119,94,.08)" radius="md" padding={3} align="center">
    **2 维 Action**
  </box>
</box>

它与 Day 4 的 `random_policy()` 最大区别在于：Actor 会利用状态信息，并且能够通过训练改变自己的行为。

## 2. Critic：状态价值评估网络

Critic 的作用是估计状态价值：

\[
V_\phi(s_t)
\]

其中 \(\phi\) 是价值网络参数。

它不直接负责控制机械臂，而是帮助训练算法评估当前策略下状态的预期回报。

注意，Actor 和 Critic 不一定必须使用完全独立的神经网络；具体实现可以共享部分特征提取网络。

## 3. Advantage：优势函数

优势函数用于衡量一个动作相对于当前策略通常表现的好坏。

理论上：

\[
\boxed{A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)}
\]

这里：

- \(Q^\pi(s,a)\)：在某状态执行某动作之后，继续按当前策略行动的预期回报；
- \(V^\pi(s)\)：当前状态下，按策略行动的平均预期回报。

如果 Advantage 为正，表示该动作相对预期更有利；如果为负，则相对不利。

实际 PPO 通常使用 GAE 等方法估计 Advantage，并不会直接计算精确的 \(Q^\pi\)。

## 4. Clip：限制策略更新幅度

PPO 的重要思想之一，是避免每次策略更新幅度过大。

它使用新旧策略对同一动作的概率比值：

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\]

并在目标函数中引入裁剪机制：

\[
L^{\mathrm{CLIP}}(\theta)=
\mathbb{E}\left[
\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

今天不要求你背下公式。

你只需要知道：

**PPO 希望让策略朝有利于提高回报的方向更新，同时通过裁剪目标减少过大的策略更新。**

它并不是严格保证每一次策略更新都不会超出固定范围，而是通过优化目标进行约束。

---

# 第五部分：检查 PPO 环境与第一周验收（第 5 小时）

## 1. 检查现有环境是否具备 PPO 训练条件

你下周要使用 Stable-Baselines3。

先检查是否已经安装：

```powershell
python -c "import stable_baselines3; print(stable_baselines3.__version__)"
```

如果没有安装，执行：

```powershell
python -m pip install "stable-baselines3[extra]"
```

然后创建 `day07_check.py`：

<CodeBlock language="python" editable>
import gymnasium as gym
import mujoco
import torch
import stable_baselines3 as sb3

print("Gymnasium:", gym.__version__)
print("MuJoCo:", mujoco.__version__)
print("PyTorch:", torch.__version__)
print("Stable-Baselines3:", sb3.__version__)
print("CUDA available:", torch.cuda.is_available())

env = gym.make("Reacher-v5")
obs, info = env.reset(seed=42)

print("\nObservation shape:", obs.shape)
print("Action space:", env.action_space)

env.close()

print("\nEnvironment check passed.")
