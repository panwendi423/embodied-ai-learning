
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(exist_ok=True)

def run_experiment(torque, damping=0.1):
    xml = f"""
    <mujoco model="one_joint_arm">
        <option timestep="0.01" gravity="0 0 0"/>

        <worldbody>
            <body name="arm">
                <joint name="hinge"
                       type="hinge"
                       axis="0 0 1"
                       damping="{damping}"/>

                <geom type="capsule"
                      fromto="0 0 0 0.5 0 0"
                      size="0.04"
                      mass="1"/>
            </body>
        </worldbody>

        <actuator>
            <motor joint="hinge"
                   gear="1"
                   ctrllimited="true"
                   ctrlrange="-1 1"/>
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    times = []
    angles = []
    velocities = []

    for step in range(300):
        data.ctrl[0] = torque
        mujoco.mj_step(model, data)

        times.append(data.time)
        angles.append(data.qpos[0])
        velocities.append(data.qvel[0])

    return np.array(times), np.array(angles), np.array(velocities)


# 实验 1：比较不同力矩
for torque in [-0.2, 0.0, 0.1, 0.2, 0.5]:
    times, angles, velocities = run_experiment(torque)

    plt.plot(
        times,
        angles,
        label=f"Torque = {torque}"
    )

    print(
        f"Torque={torque:+.1f}, "
        f"Final angle={angles[-1]:.4f}, "
        f"Final velocity={velocities[-1]:.4f}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Joint angle (rad)")
plt.title("Effect of Control Torque")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/day02_torque.png", dpi=200)
plt.close()


# 实验 2：比较不同阻尼
for damping in [0.1, 0.5, 1.0]:
    times, angles, velocities = run_experiment(
        torque=0.2,
        damping=damping
    )

    plt.plot(
        times,
        velocities,
        label=f"Damping = {damping}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Effect of Joint Damping")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/day02_damping.png", dpi=200)
plt.close()

print("\n实验完成！图片保存在 results 文件夹。")