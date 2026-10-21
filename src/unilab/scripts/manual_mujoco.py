"""Offline keyboard control for the Wheelbipe chassis in MuJoCo."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/robots/wheelbipe_v14/scene_flat.xml"
LEG_JOINTS = (
    "left_front1_joint",
    "left_rear1_joint",
    "right_front1_joint",
    "right_rear1_joint",
)
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
SPRING_JOINTS = ("left_spring2_joint", "right_spring2_joint")


@dataclass
class ManualCommand:
    """Mutable command state shared by the viewer callback and physics loop."""

    throttle: float = 0.0
    steering: float = 0.0
    posture: float = 0.0
    speed_scale: float = 1.0
    reset_requested: bool = False
    stop_requested: bool = False


def _joint_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name!r} is missing from {MODEL_PATH}")
        ids.append(joint_id)
    return np.asarray(ids, dtype=np.int32)


def _actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Actuator {name!r} is missing from {MODEL_PATH}")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=np.int32)


def _key_name(key: int) -> str:
    try:
        return chr(key).lower()
    except (TypeError, ValueError):
        return ""


def _make_key_callback(command: ManualCommand, command_lock: Lock):
    def on_key(key: int) -> None:
        name = _key_name(key)
        with command_lock:
            if name == "w":
                command.throttle = min(1.0, command.throttle + 0.1)
            elif name == "s":
                command.throttle = max(-1.0, command.throttle - 0.1)
            elif name == "a":
                command.steering = min(1.0, command.steering + 0.1)
            elif name == "d":
                command.steering = max(-1.0, command.steering - 0.1)
            elif name == "q":
                command.posture = min(1.0, command.posture + 0.1)
            elif name == "e":
                command.posture = max(-1.0, command.posture - 0.1)
            elif name == "r":
                command.reset_requested = True
            elif name == " ":
                command.throttle = 0.0
                command.steering = 0.0
            elif name == "1":
                command.speed_scale = 0.5
            elif name == "2":
                command.speed_scale = 1.0
            elif name == "3":
                command.speed_scale = 1.5
            elif key in (256, 27):  # GLFW_KEY_ESCAPE or ASCII escape.
                command.stop_requested = True

    return on_key


def _read_command(command: ManualCommand, command_lock: Lock) -> ManualCommand:
    with command_lock:
        return ManualCommand(
            throttle=command.throttle,
            steering=command.steering,
            posture=command.posture,
            speed_scale=command.speed_scale,
            reset_requested=command.reset_requested,
            stop_requested=command.stop_requested,
        )


def _set_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    command: ManualCommand,
    *,
    leg_actuators: np.ndarray,
    wheel_actuators: np.ndarray,
    spring_actuators: np.ndarray,
    leg_joints: np.ndarray,
    wheel_joints: np.ndarray,
    spring_joints: np.ndarray,
) -> None:
    """Apply the same chassis-level PD and spring behavior used by training."""
    actuator_joints = model.actuator_trnid[:, 0]
    actuator_qvel = model.jnt_dofadr[actuator_joints]
    data.ctrl[:] = -0.01 * data.qvel[actuator_qvel]

    leg_qpos = model.jnt_qposadr[leg_joints]
    leg_qvel = model.jnt_dofadr[leg_joints]
    leg_target = 0.5 * command.posture
    data.ctrl[leg_actuators] = 60.0 * (leg_target - data.qpos[leg_qpos]) - 2.0 * data.qvel[leg_qvel]

    wheel_qvel = model.jnt_dofadr[wheel_joints]
    left = command.speed_scale * (command.throttle - command.steering)
    right = command.speed_scale * (command.throttle + command.steering)
    wheel_target = np.asarray((10.0 * left, 10.0 * right), dtype=np.float64)
    data.ctrl[wheel_actuators] = 0.2 * (wheel_target - data.qvel[wheel_qvel])

    spring_qpos = model.jnt_qposadr[spring_joints]
    spring_force = 400.0 + (200.0 / 0.07) * np.maximum(0.06076 - data.qpos[spring_qpos], 0.0)
    data.ctrl[spring_actuators] = spring_force
    np.clip(
        data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1], out=data.ctrl
    )


def run(model_path: Path = MODEL_PATH) -> None:
    """Open the local MuJoCo viewer and run the keyboard-controlled chassis."""
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    leg_joints = _joint_ids(model, LEG_JOINTS)
    wheel_joints = _joint_ids(model, WHEEL_JOINTS)
    spring_joints = _joint_ids(model, SPRING_JOINTS)
    leg_actuators = _actuator_ids(model, LEG_JOINTS)
    wheel_actuators = _actuator_ids(model, WHEEL_JOINTS)
    spring_actuators = _actuator_ids(model, SPRING_JOINTS)

    command = ManualCommand()
    command_lock = Lock()
    callback = _make_key_callback(command, command_lock)
    print("Offline MuJoCo control: W/S drive, A/D turn, Q/E posture, Space stop, R reset, Esc quit")
    print("Speed: 1=slow, 2=normal, 3=fast. Each key press changes the command by 0.1.")

    with mujoco.viewer.launch_passive(model, data, key_callback=callback) as viewer:
        while viewer.is_running():
            loop_start = time.perf_counter()
            current = _read_command(command, command_lock)
            if current.stop_requested:
                break
            with viewer.lock():
                if current.reset_requested:
                    mujoco.mj_resetDataKeyframe(model, data, home_id)
                    mujoco.mj_forward(model, data)
                    with command_lock:
                        command.reset_requested = False
                _set_control(
                    model,
                    data,
                    current,
                    leg_actuators=leg_actuators,
                    wheel_actuators=wheel_actuators,
                    spring_actuators=spring_actuators,
                    leg_joints=leg_joints,
                    wheel_joints=wheel_joints,
                    spring_joints=spring_joints,
                )
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, float(model.opt.timestep) - (time.perf_counter() - loop_start)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline keyboard control for the Wheelbipe chassis."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a MuJoCo XML model")
    args = parser.parse_args(argv)
    run(args.model)


if __name__ == "__main__":
    main()
