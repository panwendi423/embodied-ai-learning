
import mujoco
import numpy as np

xml = """
<mujoco model="one_joint_arm">
    <option timestep="0.01" gravity="0 0 0"/>

    <worldbody>
        <body name="arm">
            <joint name="hinge" type="hinge"
                   axis="0 0 1" damping="0.1"/>
            <geom type="capsule"
                  fromto="0 0 0 0.5 0 0"
                  size="0.04" mass="1"/>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor" joint="hinge"
               gear="1" ctrllimited="true"
               ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Joint count:", model.njnt)
print("Actuator count:", model.nu)

positions = []

for i in range(300):
    data.ctrl[0] = 0.2
    mujoco.mj_step(model, data)

    positions.append(data.qpos[0])

print("Simulation time:", data.time)
print("Final joint position:", data.qpos[0])
print("Final joint velocity:", data.qvel[0])


# 设置施加到关节上的控制力矩
data.ctrl[0] = 0.0

# 用于记录运动过程
times = []
angles = []
velocities = []

# 运行 300 个物理仿真步
for step in range(300):

    mujoco.mj_step(model, data)

    times.append(data.time)
    angles.append(data.qpos[0])
    velocities.append(data.qvel[0])

    if step % 50 == 0:
        print(
            f"Step={step}, "
            f"Time={data.time:.2f}s, "
            f"Angle={data.qpos[0]:.4f}rad, "
            f"Velocity={data.qvel[0]:.4f}rad/s"
        )

print("\n仿真结束")
print("最终时间:", data.time)
print("最终角度:", data.qpos[0])
print("最终角速度:", data.qvel[0])
