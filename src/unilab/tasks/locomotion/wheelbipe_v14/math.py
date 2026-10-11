# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from pathlib import Path

import torch


def quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_scalar, first_vector = first[..., :1], first[..., 1:]
    second_scalar, second_vector = second[..., :1], second[..., 1:]
    first_vector, second_vector = torch.broadcast_tensors(first_vector, second_vector)
    scalar = first_scalar * second_scalar - (first_vector * second_vector).sum(-1, keepdim=True)
    vector = (
        first_scalar * second_vector
        + second_scalar * first_vector
        + torch.linalg.cross(first_vector, second_vector, dim=-1)
    )
    return torch.cat((scalar, vector), dim=-1)


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion_vector, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    twice_cross = 2 * torch.linalg.cross(quaternion_vector, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(quaternion_vector, twice_cross, dim=-1)
    )


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_from_euler(euler: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(euler / 2)
    sine = torch.sin(euler / 2)
    roll_cosine, pitch_cosine, yaw_cosine = cosine.unbind(dim=-1)
    roll_sine, pitch_sine, yaw_sine = sine.unbind(dim=-1)
    return torch.stack(
        (
            roll_cosine * pitch_cosine * yaw_cosine + roll_sine * pitch_sine * yaw_sine,
            roll_sine * pitch_cosine * yaw_cosine - roll_cosine * pitch_sine * yaw_sine,
            roll_cosine * pitch_sine * yaw_cosine + roll_sine * pitch_cosine * yaw_sine,
            roll_cosine * pitch_cosine * yaw_sine - roll_sine * pitch_sine * yaw_cosine,
        ),
        dim=-1,
    )


def yaw_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
    scalar, roll, pitch, yaw = quaternion.unbind(dim=-1)
    return torch.atan2(2 * (scalar * yaw + roll * pitch), 1 - 2 * (pitch.square() + yaw.square()))


def inverse_kinematics(length: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    links = length.new_tensor([0.1134, 0.135, 0.210])
    squared = links.square()
    offset = length.new_tensor([-6.61, 180, 29.7, 180 - 6.61 - 2 * 29.7, 29.7, 29.7])
    offset = torch.deg2rad(offset)
    triangle = torch.acos(
        (
            (squared[0] * length.square() + squared[2] * (squared[0] - squared[1]))
            / (2 * links[2] * squared[0] * length)
        ).clamp(-1, 1)
    )
    bend = torch.acos(
        (
            (squared[0] + squared[1] - squared[0] * length.square() / squared[2])
            / (2 * links[0] * links[1])
        ).clamp(-1, 1)
    )
    values = torch.stack(
        (
            angle + math.pi / 2 - triangle,
            angle + math.pi / 2 + triangle,
            bend,
            2 * math.pi - 2 * triangle - 2 * bend,
            bend,
            bend,
        ),
        dim=-1,
    )
    return (values - offset) * length.new_tensor([1, 1, -1, -1, -1, 1])


def spring_force(position: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return 400 + (600 - 400) / 0.07 * (0.06076 - position).clamp_min(0) + offset


class Delay:
    def __init__(
        self,
        num_envs: int,
        width: int,
        maximum: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.history = torch.zeros((maximum + 1, num_envs, width), dtype=dtype, device=device)
        self.lag = torch.ones(num_envs, dtype=torch.int64, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.cursor = -1
        self._rows = torch.arange(num_envs, device=device)

    def reset(self, ids: torch.Tensor, values: torch.Tensor, lag: torch.Tensor) -> None:
        self.history[:, ids] = values
        self.lag[ids] = lag
        self.count[ids] = 0

    def compute(self, values: torch.Tensor) -> torch.Tensor:
        self.cursor = (self.cursor + 1) % len(self.history)
        self.history[self.cursor] = values
        effective_lag = torch.minimum(self.lag, self.count)
        self.count += 1
        return self.history[(self.cursor - effective_lag) % len(self.history), self._rows].clone()


def reward_terms(
    command: torch.Tensor,
    linvel: torch.Tensor,
    angvel: torch.Tensor,
    gravity: torch.Tensor,
    height: torch.Tensor,
    height_command: torch.Tensor,
    wheel_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_acceleration: torch.Tensor,
    torque: torch.Tensor,
    actions: torch.Tensor,
    previous: torch.Tensor,
    before_previous: torch.Tensor,
    contact: torch.Tensor,
    terminated: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def sum_square(values: torch.Tensor) -> torch.Tensor:
        return values.square().sum(dim=-1)

    tracking = command.clone()
    tracking[command[:, 0].abs() < 0.1, 0] = 0
    tracking[command[:, 2].abs() < 0.1, 2] = 0
    pitch_cosine = (1 - gravity[:, 0].square()).clamp_min(0).sqrt()
    lin_error = tracking[:, 0] - linvel[:, 0] * pitch_cosine
    yaw_error = tracking[:, 2] - angvel[:, 2]
    height_error = height - height_command
    second_difference = actions - 2 * previous + before_previous
    orientation_gain = 2 * torch.exp(-command[:, 0].square() / 3) + 2
    fork = wheel_position[:, 0, 0] - wheel_position[:, 1, 0]
    return {
        "termination": terminated.to(command.dtype),
        "leg_joint_acc": sum_square(joint_acceleration[:, :4]),
        "leg_joint_vel": sum_square(joint_velocity[:, :4]),
        "joint_torque": sum_square(torque),
        "wheel_acc": sum_square(joint_acceleration[:, 4:]),
        "wheel_vel": sum_square(joint_velocity[:, 4:]),
        "wheel_power": (torque[:, 4:] * joint_velocity[:, 4:]).clamp_min(0).sum(dim=-1),
        "lin_vel_z": linvel[:, 2].square(),
        "ang_vel_xy": sum_square(angvel[:, :2]),
        "action_smoothness_leg": sum_square(second_difference[:, :4]),
        "action_rate": sum_square(actions - previous),
        "action_smoothness_wheel": sum_square(second_difference[:, 4:]),
        "flat_orientation_y_v": (orientation_gain * gravity[:, 0]).square(),
        "flat_orientation_x_v": (orientation_gain * gravity[:, 1]).square(),
        "flat_orientation_y_exp": torch.exp(-gravity[:, 0].square() / 0.02),
        "flat_orientation_x_exp": torch.exp(-gravity[:, 1].square() / 0.01),
        "track_lin_vel_xy": torch.exp(-lin_error.clamp(-1, 1).square() / 0.5),
        "track_lin_vel_xy_square": (lin_error * 0.25).square(),
        "track_ang_vel_z": torch.exp(-yaw_error.clamp(-0.8, 0.8).square() / 0.25),
        "track_ang_vel_z_square": (yaw_error * 0.5).square(),
        "stand_still_lin_vel": linvel[:, :2].abs().sum(dim=-1) * (command[:, 0].abs() < 0.1),
        "track_height_exp_tight": torch.exp(-height_error.clamp(-0.15, 0.15).square() / 0.001),
        "track_height_square": (height_error * 10).square(),
        "no_fork": (fork.abs() > 0.05).to(command.dtype),
        "no_fork_square": (fork * 5).square(),
        "undesired_contact": contact.to(command.dtype),
    }


class ClosedChainKinematics:
    def __init__(self, scene_path: str | Path, joint_names: Sequence[str]):
        mujoco = importlib.import_module("mujoco")

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._joint_names = tuple(joint_names)
        self._parent = tuple(int(value) for value in model.body_parentid)
        self._position = torch.tensor(model.body_pos.copy(), dtype=torch.float64)
        self._quaternion = torch.tensor(model.body_quat.copy(), dtype=torch.float64)
        self._body_joint = [-1] * model.nbody
        self._axis = torch.zeros((len(joint_names), 3), dtype=torch.float64)
        self._anchor = torch.zeros_like(self._axis)
        self._slide = [False] * len(joint_names)
        for index, name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Wheelbipe joint {name!r} is absent from the scene")
            joint_type = model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"Wheelbipe joint {name!r} must be a scalar joint")
            body_id = int(model.jnt_bodyid[joint_id])
            if self._body_joint[body_id] >= 0:
                raise ValueError("Wheelbipe closure kinematics requires one scalar joint per body")
            self._body_joint[body_id] = index
            self._axis[index] = torch.tensor(model.jnt_axis[joint_id].copy())
            self._anchor[index] = torch.tensor(model.jnt_pos[joint_id].copy())
            self._slide[index] = joint_type == mujoco.mjtJoint.mjJNT_SLIDE
        self._closure: list[tuple[torch.Tensor, torch.Tensor, tuple[int, int]]] = []
        for side in ("left", "right"):
            site_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_s2r2_joint_{index}")
                for index in (0, 1)
            ]
            if min(site_ids) < 0:
                raise ValueError(f"Wheelbipe {side} spring closure sites are absent from the scene")
            self._closure.append(
                (
                    torch.tensor(model.site_bodyid[site_ids].copy(), dtype=torch.int64),
                    torch.tensor(model.site_pos[site_ids].copy(), dtype=torch.float64),
                    (
                        self._joint_names.index(f"{side}_spring1_joint"),
                        self._joint_names.index(f"{side}_spring2_joint"),
                    ),
                )
            )
        self._regularizer = torch.eye(2, dtype=torch.float64) * 1e-9
        self._delta_limit = torch.tensor([0.5, 0.05], dtype=torch.float64)

    def closure_error(self, joint_position: torch.Tensor) -> torch.Tensor:
        if joint_position.device.type != "cpu":
            raise ValueError("Wheelbipe closure kinematics requires CPU joint positions")
        joint_position = joint_position.to(torch.float64)
        count = len(joint_position)
        positions = torch.zeros((count, len(self._parent), 3), dtype=torch.float64)
        quaternions = torch.zeros((count, len(self._parent), 4), dtype=torch.float64)
        quaternions[:, 0, 0] = 1
        for body_id in range(1, len(self._parent)):
            parent = self._parent[body_id]
            local_position = self._position[body_id].expand(count, -1).clone()
            local_quaternion = self._quaternion[body_id].expand(count, -1).clone()
            index = self._body_joint[body_id]
            if index >= 0:
                angle = joint_position[:, index]
                if self._slide[index]:
                    local_position += quat_apply(
                        local_quaternion, angle[:, None] * self._axis[index]
                    )
                else:
                    motion = torch.cat(
                        (
                            torch.cos(angle / 2)[:, None],
                            torch.sin(angle[:, None] / 2) * self._axis[index],
                        ),
                        dim=-1,
                    )
                    anchor = self._anchor[index].expand(count, -1)
                    local_position += quat_apply(
                        local_quaternion, anchor - quat_apply(motion, anchor)
                    )
                    local_quaternion = quat_mul(local_quaternion, motion)
            positions[:, body_id] = positions[:, parent] + quat_apply(
                quaternions[:, parent], local_position
            )
            quaternions[:, body_id] = quat_mul(quaternions[:, parent], local_quaternion)
        errors = []
        for body_ids, anchors, _ in self._closure:
            site_position = positions[:, body_ids] + quat_apply(
                quaternions[:, body_ids], anchors.expand(count, -1, -1)
            )
            errors.append(site_position[:, 0] - site_position[:, 1])
        return torch.stack(errors, dim=1)

    def close_springs(self, joint_position: torch.Tensor) -> None:
        if joint_position.device.type != "cpu":
            raise ValueError("Wheelbipe closure kinematics requires CPU joint positions")
        if not len(joint_position):
            return
        working_position = joint_position.to(torch.float64).clone()
        for _ in range(12):
            error = self.closure_error(working_position)
            if bool(error.abs().max() < 1e-4):
                break
            for side, (_, _, indices) in enumerate(self._closure):
                jacobian = []
                for index in indices:
                    shifted = working_position.clone()
                    shifted[:, index] += 1e-5
                    jacobian.append((self.closure_error(shifted)[:, side] - error[:, side]) / 1e-5)
                matrix = torch.stack(jacobian, dim=-1)
                normal = torch.einsum("nki,nkj->nij", matrix, matrix) + self._regularizer
                right_hand_side = torch.einsum("nki,nk->ni", matrix, error[:, side])
                delta = torch.linalg.solve(normal, right_hand_side[..., None])[..., 0]
                working_position[:, indices] -= delta.clamp(-self._delta_limit, self._delta_limit)
        joint_position.copy_(working_position)
